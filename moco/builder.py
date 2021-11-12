# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, method="mocov2"):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.method = method

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            # =======
            self.encoder_q_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k_fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)
            self.encoder_q.fc = nn.Identity()
            self.encoder_k.fc = nn.Identity()
            # =====

            # self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            # self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.encoder_q_fc.parameters(), self.encoder_k_fc.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.classifier = nn.Linear(dim_mlp, 1000)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_q, im_k, eval_method=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if eval_method:
            feats = self.encoder_q(im_q)  # queries: NxC
            classifier_output = self.classifier(feats.detach())

            return classifier_output

        if self.method == "mocov2":
            logits, labels, classifier_output = self.forward_mocov2(im_q, im_k)
            return logits, labels, classifier_output

        elif self.method == "simmoco":
            loss, logits, labels, classifier_output = self.forward_simmoco(im_q, im_k)
            return loss, logits, labels, classifier_output

        elif self.method == "simco":
            loss, classifier_output = self.forward_simco(im_q, im_k)
            return loss, classifier_output

        elif self.method == "simclr":
            loss, classifier_output = self.forward_simclr(im_q, im_k)
            return loss, classifier_output


    def forward_simclr(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feats = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q_fc(feats)
        # print(q.size())

        classifier_output = self.classifier(feats.detach())

        q1 = nn.functional.normalize(q, dim=1)


        feats = self.encoder_q(im_k)  # queries: NxC
        q = self.encoder_q_fc(feats)

        q2 = nn.functional.normalize(q, dim=1)


        loss = simclr_loss_func(q1, q2, temperature=self.T)

        return loss, classifier_output

    def forward_simco(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feats = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q_fc(feats)
        # print(q.size())

        classifier_output = self.classifier(feats.detach())

        q1 = nn.functional.normalize(q, dim=1)


        feats = self.encoder_q(im_k)  # queries: NxC
        q = self.encoder_q_fc(feats)

        q2 = nn.functional.normalize(q, dim=1)


        loss = (
                simco_loss_func(q1, q2,
                                queueintra=q2.T,
                                # queueintra=queue_intra[1],  ### it was 0 before
                                temperature=self.T)
                + simco_loss_func(q2, q1,
                                queueintra=q1.T,
                                # queueintra=queue_intra[0], 
                                temperature=self.T)
            ) / 2

        return loss, classifier_output


    def forward_simmoco(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        feats = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q_fc(feats)
        # print(q.size())


        classifier_output = self.classifier(feats.detach())


        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k_fc(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, k.T])
        l_neg.fill_diagonal_(0)

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        prob_intra = F.softmax(logits, dim=1)
        inter_intra = 1 / (1 - prob_intra[:, 0]) # setting tau inter to infinity

        loss = - torch.nn.functional.log_softmax(logits, dim=-1)[:, 0]

        loss = inter_intra.detach() * loss
        loss = loss.mean()

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        return loss, logits, labels, classifier_output



    def forward_mocov2(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feats = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q_fc(feats)
        # print(q.size())


        classifier_output = self.classifier(feats.detach())


        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k_fc(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, classifier_output
    



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def simco_loss_func(
    query: torch.Tensor,
    key: torch.Tensor, 
    # queueinter: torch.Tensor, 
    queueintra: torch.Tensor, 
    temperature=0.1
) -> torch.Tensor:
    """Computes MoCo's loss given a batch of queries from view 1, a batch of keys from view 2 and a
    queue of past elements.

    Args:
        query (torch.Tensor): NxD Tensor containing the queries from view 1.
        key (torch.Tensor): NxD Tensor containing the queries from view 2.
        queue (torch.Tensor): a queue of negative samples for the contrastive loss.
        temperature (float, optional): [description]. temperature of the softmax in the contrastive
            loss. Defaults to 0.1.

    Returns:
        torch.Tensor: MoCo loss.
    """

    # pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)
    # neg = torch.einsum("nc,ck->nk", [query, queue])
    # logits = torch.cat([pos, neg], dim=1)
    # logits /= temperature
    # targets = torch.zeros(query.size(0), device=query.device, dtype=torch.long)
    # return F.cross_entropy(logits, targets)

    # intra
    b = query.size(0)
    pos = torch.einsum("nc,nc->n", [query, key]).unsqueeze(-1)

    # Selecte the intra negative samples according the updata time, 
    neg = torch.einsum("nc,ck->nk", [query, queueintra])
    neg.fill_diagonal_(0) # moco_2queue-tau_99to99-interq_65k-intraq_256_currentkeySaveLoad

    logits = torch.cat([pos, neg], dim=1)
    logits_intra = logits / temperature

    prob_intra = F.softmax(logits_intra, dim=1)

    # inter
    # neg = torch.einsum("nc,ck->nk", [query, queueinter])

    # logits = torch.cat([pos, neg], dim=1)
    # logits_inter = logits / temperature

    # prob_inter = F.softmax(logits_inter, dim=1)

    # inter_intra = (1 - prob_inter[:, 0]) / (1 - prob_intra[:, 0])
    inter_intra = 1 / (1 - prob_intra[:, 0]) # setting tau inter to infinity

    # loss = -torch.log(prob_intra[:, 0]+1e-8)
    loss = - torch.nn.functional.log_softmax(logits_intra, dim=-1)[:, 0]

    loss = inter_intra.detach() * loss
    loss = loss.mean()

    return loss


def simclr_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        temperature (float): temperature factor for the loss. Defaults to 0.1.
        extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
            than normal across-view positives. Defaults to None.

    Returns:
        torch.Tensor: SimCLR loss.
    """

    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)



    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss
