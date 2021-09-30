#单机单卡下进行训练
python -m paddle.distributed.launch --gpus "0" main.py --n_gpu 1 --name Cifar100-ViT_B16 --gradient_accumulation_steps 32 --fp16