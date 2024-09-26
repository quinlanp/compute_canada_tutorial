#!/bin/bash
#SBATCH --job-name=train_llm        # Job name
#SBATCH --nodes=2                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --gres=gpu:4                  # Number of GPUs per node
#SBATCH --mem=0
#SBATCH --time=0-2:00:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=./slurm_out/test_llama_70b-%j.out            
#SBATCH --error=./slurm_out/test_llama_70b-%j.err
#SBATCH --account=rrg-zhu2048

module load StdEnv/2023
ml cuda python/3.11 arrow/17.0.0

pip install --upgrade pip --no-index

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

cd llama-recipes
pip install -U pip setuptools --no-index
pip install -e . --no-index

cd ..

# Environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"
echo "r$SLURM_NODEID master: $MASTER_ADDR"

# Enable for A100
export FI_PROVIDER="efa"

# Debugging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=WARN
export TORCH_NCCL_BLOCKING_WAIT=1 
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="eno8303,ib0"
export WANDB_MODE=offline

# Launch the training with torchrun
srun -c 8 \
    -N 1 \
    --mem=128000 \
    --gres=gpu:1 \
    bash -c '
    torchrun \
    --nproc-per-node=1 \
    --nnodes=1 \
    --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv-id $RDVZ_ID \
    --rdzv-backend c10d \
    ./finetune.py \
    --enable_fsdp \
    --model_name "/home/quinlanp/scratch/llama3_1-8b_instruct" \
    --dataset  \
    --dist_checkpoint_root_folder "/home/[username]/scratch/llama3-checkpoints/llama3_1_70b_instruct" \
    --project 'test_llama70b' \
    --name 'test' \
    --num_epochs 1 \
    --lr 2e-5 \
'

