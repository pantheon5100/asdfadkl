CUDA_VISIBLE_DEVICES=1 python main_lincls_our.py \
  -a resnet18 \
  --lr 30.0 \
  --batch-size 1024 \
  --pretrained ./ckpteval/mocov2_checkpoint_0040.pth.tar \
  --gpu 1 \
  --epochs 100 \
  /workspace/ssd2_4tb/data/imagenet_pytorch


