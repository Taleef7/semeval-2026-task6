#!/bin/bash
#SBATCH --job-name=t2-10seed
#SBATCH --account=pfw-cs
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --nodelist=gilbreth-j[000-001]
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/scratch/gilbreth/tamst01/clarity-semeval26
#SBATCH --output=logs/task2_10seed_%A_%a.log
#SBATCH --array=0-49%2  # 50 jobs, max 2 concurrent (2 A100-80GB nodes)

# Task 2: 10-Seed Ensemble
# 5 Folds × 10 Seeds = 50 jobs

set -e
set -u

FOLD=$((SLURM_ARRAY_TASK_ID / 10))
SEED_OFFSET=$((SLURM_ARRAY_TASK_ID % 10))
SEED=$((42 + SEED_OFFSET))

echo "============================================================"
echo "TASK 2: 10-SEED ENSEMBLE - FOLD $FOLD, SEED $SEED"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

export PROJECT_ROOT=/scratch/gilbreth/tamst01/clarity-semeval26
export HF_HOME=/scratch/gilbreth/tamst01/cache/huggingface
export PYTHONPATH=$PROJECT_ROOT/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_ROOT
source .venv/bin/activate

python src/training/train_10seed.py \
    --task 2 \
    --fold $FOLD \
    --seed $SEED \
    --model_name microsoft/deberta-xlarge \
    --lr 0.00001 \
    --label_smoothing 0.03 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --epochs 6

echo ""
echo "Fold $FOLD Seed $SEED completed at $(date)"
