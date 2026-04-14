# PFW at SemEval-2026 Task 6: Multi-Seed DeBERTa Ensembles for Political Response Clarity and Evasion Classification

[![Task](https://img.shields.io/badge/SemEval--2026-Task%206-blue)](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](latex/acl_latex.pdf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Official code and paper source for the **PFW** submission to [SemEval-2026 Task 6 (CLARITY)](https://konstantinosftw.github.io/CLARITY-SemEval-2026/). Our system placed **18/41** on Subtask 1 (Clarity) and **12/33** on Subtask 2 (Evasion). The paper has been accepted for presentation at SemEval-2026.

> **Tamsal, T. and Rusert, J. (2026).** *PFW at SemEval-2026 Task 6: Multi-Seed DeBERTa Ensembles for Political Response Clarity and Evasion Classification.* In Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026), ACL.

---

## System at a Glance

| | Subtask 1 (3-way Clarity) | Subtask 2 (9-way Evasion) |
|---|:---:|:---:|
| Architecture | DeBERTa-xlarge (900M) | DeBERTa-v3-large (304M) |
| Ensemble | 5 folds × 10 seeds = **50 models** | 5 folds × 10 seeds = **50 models** |
| Aggregation | Simple logit averaging | Simple logit averaging |
| **Macro F1 (official eval)** | **0.76** (18/41) | **0.50** (12/33) |

No LLM prompting or API access is required; the largest model is under 1B parameters and runs on a single A100 GPU.

### Baseline comparison (OOF / eval macro F1)

| System | T1 | T2 |
|---|:---:|:---:|
| Majority class | 0.248 | 0.052 |
| TF-IDF + Logistic Regression | 0.546 | 0.319 |
| ChatGPT zero-shot *(Thomas et al., 2024)* | 0.413 | 0.244 |
| DeBERTa-base fine-tuned *(Thomas et al., 2024)* | 0.441 | – |
| RoBERTa-base fine-tuned *(Thomas et al., 2024)* | 0.530 | – |
| XLNet-base fine-tuned *(Thomas et al., 2024)* | 0.518 | – |
| DeBERTa-v3-large (single seed, ours) | 0.643 ± 0.024 | 0.327 ± 0.040 |
| DeBERTa-xlarge (single seed, ours) | 0.663 ± 0.021 | – |
| **Multi-seed ensemble (ours) — official eval** | **0.76** | **0.50** |

See `latex/acl_latex.pdf` for full results and analysis.

### Key finding — the Optimization Paradox

Three independent post-hoc optimization strategies (learned ensemble weights, per-class thresholds, and hierarchical masking) each *improved* out-of-fold (OOF) macro F1 but *degraded* official evaluation scores by 0.02–0.10. We document this as an **optimization paradox**: with limited evaluation data (237 samples), *model-level* interventions (seed diversity) transfer robustly, whereas *prediction-level* interventions (post-hoc calibration) overfit OOF artifacts. See Section 5.3 of the paper.

---

## Repository Layout

```
.
├── latex/                          # Camera-ready paper source and PDF
│   ├── acl_latex.tex               # ACL-format paper
│   ├── acl_latex.pdf               # Compiled PDF
│   ├── custom.bib                  # Bibliography
│   ├── acl.sty                     # ACL template style
│   └── acl_natbib.bst              # ACL bibliography style
├── docs/
│   ├── methodology_report.md       # Extended methodology notes
│   └── paper_figures/              # Paper figures (PDF+PNG) and result tables (CSV/JSON)
├── src/
│   ├── data/                       # Dataset loading, preprocessing, CV splits
│   ├── models/                     # DeBERTa encoder-classifier architecture
│   ├── training/                   # Training loops (multi-seed, per-architecture)
│   ├── metrics/                    # Macro-F1 scorer and local evaluation harness
│   └── submission/                 # Prediction-file and submission-zip helpers
├── scripts/
│   ├── collect_task1_oof.py        # OOF-logit collection (Task 1, xlarge)
│   ├── collect_v3large_oof.py      # OOF-logit collection (both tasks, v3-large)
│   ├── build_oof_logits.py         # OOF-matrix utilities
│   ├── generate_final_ensemble.py  # End-to-end inference → submission
│   ├── generate_simple_ensemble.py # Reference logit-averaging ensemble
│   ├── eval_task1_predictions.py   # Local T1 scorer
│   ├── eval_task2_predictions.py   # Local T2 scorer
│   ├── local_eval.py               # Combined local evaluator
│   ├── paper_baselines.py          # Reproduce majority / TF-IDF baselines
│   ├── paper_analysis.py           # Regenerate figures and result tables
│   └── slurm/                      # SLURM job scripts for an HPC workflow
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

Everything outside this tree (raw data, checkpoints, OOF logits, training logs, submission archives) is produced by the training pipeline and gitignored.

---

## Reproduction

### Environment

- Python 3.9+ with PyTorch 2.0+ (CUDA 12.1)
- 1× NVIDIA A100-80GB recommended for DeBERTa-xlarge; a 24 GB card is sufficient for DeBERTa-v3-large
- Estimated compute for the full 100-model campaign: ~50 GPU-hours

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Data

The QEvasion dataset (Thomas et al., 2024) is loaded from the HuggingFace Hub:

```python
from datasets import load_dataset
dataset = load_dataset("ailsntua/QEvasion")
```

The official SemEval-2026 evaluation set is distributed via the task organizers and is not included in this repository.

### Training pipeline

**1. Generate stratified 5-fold splits** (writes to `artifacts/splits/`):

```bash
python src/data/splits.py
```

**2. Train one fold × one seed** — primary entry point:

```bash
python src/training/train_10seed.py \
    --task 1 --fold 0 --seed 42 \
    --model_name microsoft/deberta-xlarge \
    --epochs 3 --lr 2e-5
```

For the full 50-model ensemble per architecture, submit the provided SLURM arrays:

```bash
sbatch scripts/slurm/task1_10seed.sbatch    # Task 1, all (fold × seed) combos
sbatch scripts/slurm/task2_10seed.sbatch    # Task 2, all (fold × seed) combos
```

**3. Collect OOF logits:**

```bash
python scripts/collect_task1_oof.py          # xlarge → Task 1
python scripts/collect_v3large_oof.py        # v3-large → Tasks 1 & 2
```

**4. Generate ensemble predictions:**

```bash
python scripts/generate_simple_ensemble.py   # logit averaging (no calibration)
```

**5. Evaluate locally:**

```bash
python scripts/local_eval.py --task 1 --prediction_file submissions/task1_prediction
python scripts/local_eval.py --task 2 --prediction_file submissions/task2_prediction
```

### Regenerating paper artifacts

```bash
python scripts/paper_baselines.py    # majority-class and TF-IDF baselines
python scripts/paper_analysis.py     # figures and result tables in docs/paper_figures/
```

### Building the paper

```bash
cd latex
pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex
```

The camera-ready PDF compiles cleanly under the ACL 2026 style, passes `aclpubcheck --paper_type long`, and fits the 6-page main-body limit.

---

## Task Description

SemEval-2026 Task 6 ([CLARITY](https://konstantinosftw.github.io/CLARITY-SemEval-2026/)) addresses political question evasion detection over the [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) dataset:

- **Subtask 1**: 3-way clarity classification — *Clear Reply* / *Ambivalent* / *Clear Non-Reply*
- **Subtask 2**: 9-way evasion-type classification — *Explicit*, *Dodging*, *Deflection*, *Implicit*, *General*, *Partial/half-answer*, *Claims ignorance*, *Declining to answer*, *Clarification*

Both subtasks are scored by macro F1 over a 237-sample held-out evaluation set.

---

## Citation

```bibtex
@inproceedings{tamsal2026pfw,
  title     = {{PFW} at {SemEval}-2026 Task 6: Multi-Seed {DeBERTa} Ensembles for Political Response Clarity and Evasion Classification},
  author    = {Tamsal, Taleef and Rusert, Jonathan},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear}
}
```

---

## License

The code in this repository is released under the [MIT License](LICENSE). The QEvasion dataset is governed by its own [license terms](https://huggingface.co/datasets/ailsntua/QEvasion) and is not redistributed here.

## Acknowledgments

We thank the CLARITY task organizers for the QEvasion dataset and the shared task. Computational resources were provided by Purdue University Fort Wayne through the Gilbreth HPC cluster.
