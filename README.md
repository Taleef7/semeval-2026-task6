# PFW at SemEval-2026 Task 6: Multi-Seed DeBERTa Ensembles for Political Response Clarity and Evasion Classification

This repository contains the code for our system submission to [SemEval-2026 Task 6 (CLARITY)](https://konstantinosftw.github.io/CLARITY-SemEval-2026/), which addresses the classification of response clarity and evasion techniques in political interview question-answer pairs.

## System Overview

Our approach fine-tunes DeBERTa-xlarge (900M) and DeBERTa-v3-large (304M) with a multi-seed ensemble strategy:
- **5-fold cross-validation** with **10 random seeds** yields **50 models** per architecture
- Predictions are combined via **simple logit averaging**
- No LLM prompting or API calls required — runs on a single GPU

### Results

| System | Subtask 1 (Clarity) | Subtask 2 (Evasion) |
|--------|:---:|:---:|
| Majority class baseline | 0.248 | 0.052 |
| TF-IDF + Logistic Regression | 0.546 | 0.319 |
| DeBERTa-v3-large (single model) | 0.643 ± 0.024 | 0.327 ± 0.040 |
| DeBERTa-xlarge (single model) | 0.663 ± 0.021 | — |
| **Multi-seed ensemble (ours)** | **0.76** (18/41) | **0.50** (12/33) |

Macro F1 scores. Rank on official leaderboard in parentheses.

## Project Structure

```
├── src/
│   ├── training/           # Training scripts
│   │   ├── train_10seed.py          # Primary: unified multi-seed training (Task 1 & 2)
│   │   ├── train_v3large_task1.py   # v3-large Task 1 training
│   │   ├── train_v3large_task2.py   # v3-large Task 2 training
│   │   ├── train_task1_xlarge.py    # xlarge Task 1 training
│   │   └── train_utils.py           # Shared training utilities
│   ├── models/
│   │   └── encoder_classifier.py    # DeBERTa encoder-classifier architecture
│   ├── data/
│   │   ├── load_dataset.py          # HuggingFace data loading
│   │   ├── preprocess.py            # Text preprocessing & label normalization
│   │   ├── splits.py                # GroupKFold CV split generation
│   │   └── stratified_group_kfold.py
│   ├── metrics/
│   │   ├── compute_metrics.py       # Macro F1 computation
│   │   └── local_test_scorer.py     # Local evaluation scorer
│   └── submission/
│       ├── make_prediction_file.py  # Single-model predictions
│       ├── make_prediction_file_ensemble.py  # Ensemble predictions
│       └── zip_submission.py        # Submission packaging
├── scripts/
│   ├── generate_final_ensemble.py   # Final ensemble inference pipeline
│   ├── generate_simple_ensemble.py  # Simple logit-averaging ensemble
│   ├── paper_baselines.py           # Reproduce paper baselines
│   ├── paper_analysis.py            # Generate paper figures and tables
│   ├── build_oof_logits.py          # Build OOF logit matrices
│   ├── collect_task1_oof.py         # OOF collection for Task 1
│   ├── collect_v3large_oof.py       # OOF collection for v3-large
│   ├── eval_task1_predictions.py    # Task 1 evaluation
│   ├── eval_task2_predictions.py    # Task 2 evaluation
│   ├── local_eval.py               # Local evaluation harness
│   └── slurm/                       # SLURM job scripts for HPC
├── latex/                           # Paper source (ACL format)
├── docs/                            # Documentation and paper figures
└── requirements.txt
```

## Reproduction

### Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- 1x NVIDIA A100 (80GB) recommended; runs on any GPU with >= 24GB

```bash
pip install -r requirements.txt
```

### Data

The QEvasion dataset is loaded automatically from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("ailsntua/QEvasion")
```

### Training

**Step 1: Generate cross-validation splits**
```bash
python src/data/splits.py
```

**Step 2: Train multi-seed models (example: Task 1, xlarge)**
```bash
# Single fold + seed
python src/training/train_10seed.py \
    --task 1 --fold 0 --seed 42 \
    --model_name microsoft/deberta-xlarge \
    --epochs 6 --lr 1e-5 --label_smoothing 0.03

# Or submit all fold x seed combinations via SLURM
sbatch scripts/slurm/task1_10seed.sbatch
```

**Step 3: Collect OOF logits**
```bash
python scripts/collect_task1_oof.py
```

**Step 4: Generate ensemble predictions**
```bash
python scripts/generate_simple_ensemble.py
```

### Evaluation

```bash
python scripts/local_eval.py --task 1 --prediction_file submissions/task1_prediction
python scripts/local_eval.py --task 2 --prediction_file submissions/task2_prediction
```

## Task Description

SemEval-2026 Task 6 ([CLARITY](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)) addresses political question evasion detection:

- **Subtask 1**: Classify responses into 3 clarity levels (*Clear Reply*, *Ambivalent*, *Clear Non-Reply*)
- **Subtask 2**: Classify into 9 fine-grained evasion types (*Explicit*, *Dodging*, *Deflection*, etc.)

Both are evaluated using macro F1 on the [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) (3,448 training / 237 evaluation instances).

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{tamsal2026pfw,
  title     = {{PFW} at {SemEval}-2026 Task 6: Multi-Seed {DeBERTa} Ensembles for Political Response Clarity and Evasion Classification},
  author    = {Tamsal, Taleef},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

## License

This project is released for research purposes. The QEvasion dataset is subject to its own [license terms](https://huggingface.co/datasets/ailsntua/QEvasion).
