#!/bin/bash
#SBATCH -p gpu17
#SBATCH --gres gpu:2
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 2
#SBATCH -t 0-01:00:0

# activate Conda environment
source ~/.bashrc
conda activate ldm-new

cd ..

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# print start time
echo $(date)
echo $SLURM_NTASKS
echo $MASTER_ADDR
echo $MASTER_PORT

# start command
srun python3 analysis.py
