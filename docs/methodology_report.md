# SemEval-2026 Task 6: Clarity Classification Methodology Report

> This document supplements the [camera-ready paper](../latex/acl_latex.pdf) with developer-facing notes on the training pipeline, submission history, and the post-hoc optimization strategies that did not transfer from OOF to evaluation. For the formal methodology, error analysis, and discussion of the *optimization paradox*, see the paper.

## Competition Overview

**Task**: SemEval-2026 Task 6 - Clarity Classification  
**Subtask 1**: Clarity Level Classification (3-class: Clear Reply, Ambivalent, Clear Non-Reply)  
**Subtask 2**: Evasion Type Classification (9-class: Explicit, Dodging, Claims ignorance, etc.)  
**Evaluation Metric**: Macro F1 Score  
**Team Best Scores**: 
- **Subtask 1: 0.76** (Submission ID: 489568, Jan 15, 2026)
- **Subtask 2: 0.50** (Submission ID: 489570, Jan 15, 2026)

---

## Final Submission Results

### Submission History


| ID | Task | Date | Score | Notes |
|----|------|------|-------|-------|
| 489568 | Task 1 | 2026-01-15 12:09 | **0.76** | [BEST] Best |
| 486446 | Task 1 | 2026-01-13 18:06 | 0.54 | Baseline |
| 497152 | Task 1 | 2026-01-20 11:23 | 0.74 | Over-optimized |


| ID | Task | Date | Score | Notes |
|----|------|------|-------|-------|
| 489570 | Task 2 | 2026-01-15 12:09 | **0.50** | [BEST] Best |
| 486447 | Task 2 | 2026-01-13 18:06 | 0.50 | Tied best |
| 492081 | Task 2 | 2026-01-17 11:11 | 0.40 | Over-optimized |

---

## Best Performing Approach (0.76 / 0.50)

### Architecture

**Base Models**:
- DeBERTa-xlarge (microsoft/deberta-xlarge) - 900M parameters
- DeBERTa-v3-large (microsoft/deberta-v3-large) - 304M parameters

**Ensemble Strategy**:
- 5-fold cross-validation
- 10 random seeds per fold (50 total models per architecture)
- Simple average aggregation across all models

### Training Configuration

```python
# Hyperparameters (Best Performing)
model_name = "microsoft/deberta-xlarge"  # Task 1 primary
learning_rate = 2e-5
batch_size = 4
gradient_accumulation_steps = 8  # Effective batch size: 32
max_length = 512
epochs = 3
warmup_ratio = 0.1
weight_decay = 0.01
optimizer = "AdamW"
scheduler = "linear_with_warmup"
```

### Data Processing

```python
# Input format
text = f"Question: {row['question']}\nAnswer: {row['interview_answer']}"

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoding = tokenizer(text, max_length=512, padding='max_length', truncation=True)
```

### Model Architecture

```python
class Task1Classifier(nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits
```

### Ensemble Inference

```python
# Simple average ensemble (BEST APPROACH)
ensemble_logits = np.zeros((n_samples, n_classes))

for model in models:
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        ensemble_logits += logits.cpu().numpy()

ensemble_logits /= len(models)
predictions = ensemble_logits.argmax(axis=1)
```

---

## What Worked

### 1. Large Pre-trained Models
- DeBERTa-xlarge significantly outperformed smaller models
- The enhanced disentangled attention mechanism was effective for nuanced text classification

### 2. Multi-Seed Ensembling
- 10 seeds × 5 folds = 50 models provided robust predictions
- Reduced variance and improved generalization

### 3. Simple Aggregation
- **Simple average** of logits outperformed weighted combinations
- No post-hoc calibration needed for best results

### 4. Standard Training Recipe
- Learning rate 2e-5, 3 epochs, linear warmup
- No special augmentation or regularization

---

## What Did NOT Work

### 1. Optuna Weight Optimization
- Optimized architecture weights (72.5% xlarge, 27.5% v3) on OOF
- **Result**: OOF improved (+0.003) but eval dropped (-0.02)
- **Lesson**: Overfitting to OOF distribution

### 2. Class-Specific Thresholds
- Calibrated per-class decision thresholds
- **Result**: Further degraded eval performance
- **Lesson**: OOF->Eval distribution shift

### 3. Learned Hierarchical Masking (Task 2)
- Learned 3×9 interaction matrix between Task 1 and Task 2
- **Result**: 0.40 vs baseline 0.50 (-0.10!)
- **Lesson**: Introduced cascading errors

### 4. Mega-Ensemble (233+ models)
- Including all available checkpoints
- **Result**: No improvement, possibly added noise
- **Lesson**: More models ≠ better when quality varies

---

## Key Lessons Learned

### 1. OOF ≠ Evaluation
> "Improvements on OOF data do not guarantee improvements on held-out evaluation data."

Our most optimized approach showed +0.01 F1 on OOF but -0.02 on eval. The distribution shift between training folds and evaluation set was significant.

### 2. Simplicity Beats Complexity
> "Simple ensemble averaging outperformed sophisticated learned weighted combinations."

The best submission used no post-hoc optimization - just train good models and average their predictions.

### 3. Hierarchical Conditioning is Risky
> "Task 2 predictions conditioned on Task 1 can cascade errors."

When Task 1 predictions are wrong, the learned masking matrix amplifies errors in Task 2.

### 4. Trust Your Baselines
> "A well-tuned baseline often outperforms over-engineered solutions."

The Jan 15 submission was relatively simple but robust. Additional "optimizations" hurt generalization.

---

## Reproducibility

### Environment
```bash
Python 3.10.12
PyTorch 2.1.0
Transformers 4.36.0
CUDA 12.1
Hardware: NVIDIA A100-80GB
```

### Training Commands
```bash
# Task 1: DeBERTa-xlarge ensemble
sbatch scripts/slurm/train_xlarge_task1.sbatch  # 50 jobs (5 folds × 10 seeds)

# Task 2: DeBERTa-v3-large ensemble
sbatch scripts/slurm/train_v3large_task2.sbatch  # 50 jobs
```

### Inference
```bash
python scripts/eval_task1_predictions.py
python scripts/eval_task2_predictions.py
```

---

## File Structure

```
clarity-semeval26/
├── src/
│   ├── training/
│   │   ├── train_xlarge_task1.py
│   │   ├── train_v3large_task1.py
│   │   └── train_v3large_task2.py
│   └── data/
│       └── dataset.py
├── scripts/
│   ├── eval_task1_predictions.py
│   ├── eval_task2_predictions.py
│   ├── collect_task1_oof.py
│   └── slurm/
│       ├── train_xlarge_task1.sbatch
│       └── train_v3large_task2.sbatch
├── runs/
│   ├── task1_xlarge_fold*_seed*/
│   └── task2_v3large_fold*_seed*/
├── artifacts/
│   └── oof/
│       ├── task1_10seed_oof.npz
│       └── task2_v3large_oof.npz
└── submissions/
    └── [best submission files]
```

---

## Ablation Study Summary

| Configuration | Task 1 F1 | Task 2 F1 | Notes |
|--------------|-----------|-----------|-------|
| xlarge only (simple avg) | **0.76** | - | [BEST] Best T1 |
| v3-large only | 0.74 | 0.50 | Weaker T1 |
| xlarge + v3 weighted | 0.74 | 0.40 | Over-optimized |
| + threshold calibration | 0.74 | 0.40 | No improvement |
| + hierarchical masking | - | 0.40 | [FAIL] Hurt T2 |
| Simple ensemble (baseline) | 0.76 | **0.50** | [BEST] Best overall |

---

## Recommendations for Future Work

1. **Validate on held-out dev set** before optimizing on OOF
2. **Avoid cascading dependencies** between subtasks
3. **Use early stopping** based on validation loss, not OOF F1
4. **Consider calibration** only with proper validation data
5. **Test distribution shift** between OOF and expected eval

---

## Citation

```bibtex
@inproceedings{semeval2026-task6-clarity,
  title     = {DeBERTa Ensemble for Clarity Classification in Political Interviews},
  author    = {[Author Names]},
  booktitle = {Proceedings of SemEval-2026},
  year      = {2026}
}
```

---

## Appendix: Training Logs

### DeBERTa-xlarge Task 1 Training
- Total models trained: 50 (5 folds × 10 seeds)
- Average training time: ~45 minutes per model
- Average OOF F1: 0.6794
- Evaluation F1: **0.76**

### DeBERTa-v3-large Task 2 Training
- Total models trained: 50 (5 folds × 10 seeds)
- Average training time: ~15 minutes per model
- Average OOF F1: 0.3727
- Evaluation F1: **0.50**

### Optimization Experiments (Jan 17)
- Optuna trials: 1500
- Best OOF improvement: +0.01
- Actual eval degradation: -0.02 (Task 1), -0.10 (Task 2)
- **Conclusion**: Over-optimization caused overfitting

---

*Document generated: January 20, 2026*
