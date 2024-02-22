#!/usr/bin/bash -l

#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=4
#SBATCH --output=output_slurm/training/ImageNet_out_DDPM.%j
#SBATCH --error=output_slurm/training/ImageNet_err_DDPM.%j

source modules.env

cp -r /mnt/home/gkrawezik/ceph/AI_DATASETS/ImageNet/2012/nano_imagenet /tmp/imagenet/
pushd /tmp/imagenet/
mkdir val train
cd train/
unzip -qq ../train.zip
cd ../val
unzip -qq ../val.zip
popd

torchrun --standalone --nproc_per_node=4 train.py --n_devices 4 --dataset_choice=ImageNet \
    --optimizer=AdamW --learning_rate 0.005 --weight_decay 0 \
    --diffusion_steps 1000 --max_epoch 100 --batch_size 128  \
    --wandb_group_name=afterHP_imagenet_v3 --wandb=True --enable_ckpt=True
    
