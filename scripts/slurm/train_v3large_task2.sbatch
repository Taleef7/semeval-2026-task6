#!/bin/bash
#SBATCH --job-name=v3-t2-%a
#SBATCH --account=pfw-cs
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --array=0-49%8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/scratch/gilbreth/tamst01/clarity-semeval26
#SBATCH --output=logs/v3_task2_%A_%a.log

set -e
set -u

echo "============================================================"
echo "DeBERTa-v3-large Task 2 Training"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

export PROJECT_ROOT=/scratch/gilbreth/tamst01/clarity-semeval26
export HF_HOME=/scratch/gilbreth/tamst01/cache/huggingface
export PYTHONPATH=$PROJECT_ROOT/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_ROOT
source .venv/bin/activate

# Map array task ID to (fold, seed)
FOLD=$((SLURM_ARRAY_TASK_ID / 10))
SEED=$((42 + SLURM_ARRAY_TASK_ID % 10))

echo "Training: Fold $FOLD, Seed $SEED"

python src/training/train_v3large_task2.py \
    --fold $FOLD \
    --seed $SEED

echo ""
echo "Completed at $(date)"
