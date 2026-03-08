#!/bin/bash
#SBATCH --job-name=oof-collect
#SBATCH --account=pfw-cs
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=2:00:00
#SBATCH --chdir=/scratch/gilbreth/tamst01/clarity-semeval26
#SBATCH --output=logs/oof_collection_%j.log

set -e
set -u

echo "============================================================"
echo "COLLECTING OOF PREDICTIONS"
echo "============================================================"

export PROJECT_ROOT=/scratch/gilbreth/tamst01/clarity-semeval26
export HF_HOME=/scratch/gilbreth/tamst01/cache/huggingface
export PYTHONPATH=$PROJECT_ROOT/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_ROOT
source .venv/bin/activate

python scripts/collect_task1_oof.py

echo ""
echo "Completed at $(date)"
