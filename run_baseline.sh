python main_moco.py \
  -a resnet18 \
  --lr 0.015 \
  --epochs 100 \
  --method simmoco \
  --batch-size 128 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  /workspace/ssd2_4tb/data/imagenet_pytorch

