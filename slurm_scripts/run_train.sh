#!/bin/bash
#SBATCH -p gpu17
#SBATCH --gres gpu:2
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 2
#SBATCH -t 0-08:00:0
#SBATCH -o train_output_%j.log
#SBATCH -e train_error_%j.log

# activate Conda environment
source ~/.bashrc
conda activate ldm

cd ..

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# print start time
echo $(date)
echo $SLURM_NTASKS
echo $MASTER_ADDR
echo $MASTER_PORT

# start command
srun python3 main_mar.py \
    --batch_size 64 \
    --data_path /path/to/imagenet-1k \
    --vae_embed_dim 64 \
    --vae_type vavae-high-low \
    --diffloss_d 3 \
    --diffloss_w 1024 \
    --epochs 400 \
    --warmup_epochs 100 \
    --blr 1.0e-4 \
    --diffusion_batch_mul 4 \
    --world_size $SLURM_NTASKS \
    --num_workers 4 \
    --output_dir output_dir \
    --resume output_dir \
    --save_last_freq 1
