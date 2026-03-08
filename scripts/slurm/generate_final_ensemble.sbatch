#!/bin/bash
#SBATCH --job-name=final-ensemble
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --chdir=/scratch/gilbreth/tamst01/clarity-semeval26
#SBATCH --output=logs/final_ensemble_%j.log

set -e
set -u

echo "============================================================"
echo "Generating Final Optimized Ensemble Predictions"
echo "============================================================"

export PROJECT_ROOT=/scratch/gilbreth/tamst01/clarity-semeval26
export HF_HOME=/scratch/gilbreth/tamst01/cache/huggingface
export PYTHONPATH=$PROJECT_ROOT/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_ROOT
source .venv/bin/activate

python scripts/generate_final_ensemble.py

echo ""
echo "Completed at $(date)"
