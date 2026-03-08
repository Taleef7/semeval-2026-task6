#!/bin/bash
#SBATCH --job-name=collect-oof
#SBATCH --account=pfw-cs
#SBATCH --partition=a100-80gb
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --chdir=/scratch/gilbreth/tamst01/clarity-semeval26
#SBATCH --output=logs/collect_v3large_oof_%j.log

set -e
set -u

echo "============================================================"
echo "Collecting v3-large OOF predictions"
echo "============================================================"

export PROJECT_ROOT=/scratch/gilbreth/tamst01/clarity-semeval26
export HF_HOME=/scratch/gilbreth/tamst01/cache/huggingface
export PYTHONPATH=$PROJECT_ROOT/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

cd $PROJECT_ROOT
source .venv/bin/activate

python scripts/collect_v3large_oof.py

echo ""
echo "Completed at $(date)"
