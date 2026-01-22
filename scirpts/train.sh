
# in 07 base
torchrun --nproc_per_node $SENSECORE_ACCELERATE_DEVICE_COUNT --nnodes $SENSECORE_PYTORCH_NNODES \
    --node_rank $SENSECORE_PYTORCH_NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    src/train_base.py task_name=base trainer.check_val_every_n_epoch=10 \
    trainer=ddp trainer.devices=auto \
    data.args.batch_size_per_device=20

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node 2 \
    src/train_base.py task_name=base trainer.check_val_every_n_epoch=10 \
    trainer=ddp trainer.devices=auto \
    data.args.batch_size_per_device=8

CUDA_VISIBLE_DEVICES=0,1,2,3 screen -U python src/train_base.py task_name=base trainer.check_val_every_n_epoch=10 \
    trainer=ddp trainer.devices=4 \
    data.args.batch_size_per_device=20 \
    ckpt_path=/mnt/sharedata/ssd_large/users/guohl/ai4sci/ForceFM2/workdir/base/runs/2025-12-05_16-47-40/checkpoints/epoch_1869.ckpt 

CUDA_VISIBLE_DEVICES=0,1,2,3 screen -U python src/train_guidance.py task_name=guidance trainer.check_val_every_n_epoch=10 \
    trainer=ddp trainer.devices=4 \
    trainer.max_epochs=400 data.args.batch_size_per_device=8

 # Inf 
CUDA_VISIBLE_DEVICES=4 screen -U python inference.py --out_dir results/try
CUDA_VISIBLE_DEVICES=1 screen -U python inference.py --out_dir results/gt

# Eval
python evaluate_files.py --results_path results/base2_1919

# in 04 fm
CUDA_VISIBLE_DEVICES=3,4,5,6 screen -U python src/train_base.py task_name=base trainer.check_val_every_n_epoch=10 \
    trainer=ddp trainer.devices=4 paths.data_dir=/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020 \
    data.args.batch_size_per_device=8 \
    ckpt_path=/mnt/sharedata/ssd_large/users/guohl/ai4sci/ForceFM2/workdir/basemodel/epoch_1849.ckpt

CUDA_VISIBLE_DEVICES=3 screen -U  python inference.py --out_dir results/base --inference_steps 16 --actual_steps 15

CUDA_VISIBLE_DEVICES=2 screen -U  python inference.py --out_dir results/base2_1989_15 --ckpt epoch_1989.ckpt --inference_steps 16 --actual_steps 15
CUDA_VISIBLE_DEVICES=3 screen -U  python inference.py --out_dir results/base2_1999_15 --ckpt epoch_1999.ckpt --inference_steps 16 --actual_steps 15

CUDA_VISIBLE_DEVICES=3 screen -U  python inference.py --out_dir results/base2_689 --ckpt epoch_689.ckpt
# CUDA_VISIBLE_DEVICES=0,1,2,3 screen -U python src/train_base.py task_name=pretrain trainer.check_val_every_n_epoch=10 \
#     trainer=ddp trainer.devices=4 paths.data_dir=/mnt/sharedata/ssd_large/users/guohl/datasets/ai4sci/pdbbind2020 \
#     data.args.batch_size_per_device=20 \
#     ckpt_path=/mnt/sharedata/ssd_large/users/guohl/ai4sci/ForceFM2/workdir/basemodel/epoch_1329.ckpt