#!/usr/bin/env python3
"""
Qualitative Error Analysis + Baseline Comparison Script
SemEval-2026 Task 6: CLARITY

Generates:
  1. Qualitative error examples (T1 and T2)
  2. TF-IDF + LogReg baseline for comparison
  3. Majority-class baseline
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

OUT = Path("docs/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

T1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
T2_LABELS = ["Claims ignorance", "Clarification", "Declining to answer",
             "Deflection", "Dodging", "Explicit", "General",
             "Implicit", "Partial/half-answer"]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading training data …")
df = pd.read_csv("data/qevasion_train.csv")
print(f"  {len(df)} samples loaded")

# Input text: Question + Answer
df["text"] = "Question: " + df["interview_question"].fillna("") + "\nAnswer: " + df["interview_answer"].fillna("")

# Encode labels
le_t1 = LabelEncoder()
le_t2 = LabelEncoder()
t1_int = le_t1.fit_transform(df["clarity_label"])
t2_int = le_t2.fit_transform(df["evasion_label"])

print(f"  T1 classes: {le_t1.classes_}")
print(f"  T2 classes: {le_t2.classes_}")

# Load OOF predictions
t1x = np.load("artifacts/oof/task1_10seed_oof.npz")
t2v = np.load("artifacts/oof/task2_v3large_oof.npz")

t1_logits  = t1x["logits"]          # (3448, 3, 50)
t1_labels  = t1x["labels"]          # (3448,)
t1_mask    = t1x["val_mask"]        # (3448,)
t2_logits  = t2v["logits"]          # (3448, 9, 50)
t2_labels  = t2v["ground_truth"]    # (3448,)
t2_mask    = t2v["mask"]            # (3448, 50)

# T1 ensemble predictions
t1_avg  = t1_logits[np.where(t1_mask)[0]].mean(axis=2)
t1_pred = t1_avg.argmax(axis=1)
t1_true = t1_labels[np.where(t1_mask)[0]]
t1_idx  = np.where(t1_mask)[0]   # Original training indices

# T2 ensemble predictions
weighted2 = (t2_logits * t2_mask[:, np.newaxis, :]).sum(axis=2)
counts2   = t2_mask.sum(axis=1, keepdims=True).clip(min=1)
avg2      = weighted2 / counts2
t2_idx_mask = np.where(t2_mask.any(axis=1))[0]
t2_pred = avg2[t2_idx_mask].argmax(axis=1)
t2_true = t2_labels[t2_idx_mask]

# ════════════════════════════════════════════════════════════════════════════
# 1. QUALITATIVE ERROR EXAMPLES
# ════════════════════════════════════════════════════════════════════════════

print("\n[1] Extracting qualitative error examples …")

# T1 errors
t1_error_mask = t1_pred != t1_true
t1_error_local = np.where(t1_error_mask)[0]   # indices within val subset
t1_error_orig  = t1_idx[t1_error_local]         # indices in df

# Confidence (softmax max prob)
t1_probs = np.exp(t1_avg) / np.exp(t1_avg).sum(axis=1, keepdims=True)
t1_conf  = t1_probs.max(axis=1)

# Sort by confidence (most confident errors = most interesting)
sort_by_conf = t1_error_local[np.argsort(t1_conf[t1_error_local])[::-1]]

t1_examples = []
seen_pairs = {}
for li in sort_by_conf:
    oi = t1_idx[li]
    true_lbl = T1_LABELS[t1_true[li]]
    pred_lbl = T1_LABELS[t1_pred[li]]
    pair_key = (true_lbl, pred_lbl)
    if pair_key not in seen_pairs:
        seen_pairs[pair_key] = 0
    if seen_pairs[pair_key] >= 2:
        continue
    seen_pairs[pair_key] += 1
    if oi >= len(df):
        continue
    row = df.iloc[oi]
    t1_examples.append({
        "idx": int(oi),
        "question": str(row["interview_question"])[:400],
        "answer": str(row["interview_answer"])[:500],
        "true_label": true_lbl,
        "pred_label": pred_lbl,
        "confidence": float(t1_conf[li]),
        "president": str(row.get("president", ""))
    })
    if len(t1_examples) >= 12:
        break

# T2 errors
t2_error_mask = t2_pred != t2_true
t2_error_local = np.where(t2_error_mask)[0]
t2_error_orig  = t2_idx_mask[t2_error_local]

t2_probs = np.exp(avg2[t2_idx_mask]) / np.exp(avg2[t2_idx_mask]).sum(axis=1, keepdims=True)
t2_conf  = t2_probs.max(axis=1)
sort_by_conf2 = t2_error_local[np.argsort(t2_conf[t2_error_local])[::-1]]

t2_examples = []
seen_pairs2 = {}
for li in sort_by_conf2:
    oi = t2_idx_mask[li]
    true_lbl = T2_LABELS[t2_true[li]]
    pred_lbl = T2_LABELS[t2_pred[li]]
    pair_key = (true_lbl, pred_lbl)
    if pair_key not in seen_pairs2:
        seen_pairs2[pair_key] = 0
    if seen_pairs2[pair_key] >= 1:
        continue
    seen_pairs2[pair_key] += 1
    if oi >= len(df):
        continue
    row = df.iloc[oi]
    t2_examples.append({
        "idx": int(oi),
        "question": str(row["interview_question"])[:400],
        "answer": str(row["interview_answer"])[:500],
        "true_label": true_lbl,
        "pred_label": pred_lbl,
        "confidence": float(t2_conf[li]),
        "president": str(row.get("president", ""))
    })
    if len(t2_examples) >= 15:
        break

with open(OUT / "qualitative_errors.json", "w") as f:
    json.dump({"t1_errors": t1_examples, "t2_errors": t2_examples}, f, indent=2, ensure_ascii=False)

print(f"  Saved {len(t1_examples)} T1 and {len(t2_examples)} T2 qualitative examples")
print("\n  ── T1 Top Errors ──")
for ex in t1_examples[:4]:
    print(f"\n  [TRUE: {ex['true_label']}  ->  PRED: {ex['pred_label']}]  conf={ex['confidence']:.3f}")
    print(f"  Q: {ex['question'][:150]}")
    print(f"  A: {ex['answer'][:200]}")

# ════════════════════════════════════════════════════════════════════════════
# 2. BASELINE COMPARISONS (TF-IDF + Logistic Regression)
# ════════════════════════════════════════════════════════════════════════════

print("\n[2] Running baseline comparisons …")

texts  = df["text"].values
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_baseline(labels, task_name, label_names):
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True)
    lr    = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)

    all_preds, all_true = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, labels)):
        X_tr = tfidf.fit_transform([texts[i] for i in tr_idx])
        X_val = tfidf.transform([texts[i] for i in val_idx])
        lr.fit(X_tr, labels[tr_idx])
        preds = lr.predict(X_val)
        all_preds.extend(preds)
        all_true.extend(labels[val_idx])
        f1 = f1_score(labels[val_idx], preds, average="macro")
        print(f"    Fold {fold+1}: {task_name} F1 = {f1:.4f}")

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)
    macro_f1 = f1_score(all_true, all_preds, average="macro")
    print(f"  -> {task_name} TF-IDF+LR OOF Macro F1: {macro_f1:.4f}")

    report = classification_report(all_true, all_preds, target_names=label_names, output_dict=True)
    return macro_f1, report

# Majority class baseline
def majority_baseline(labels, task_name):
    from collections import Counter
    majority = Counter(labels).most_common(1)[0][0]
    preds = np.full_like(labels, majority)
    f1 = f1_score(labels, preds, average="macro")
    print(f"  -> {task_name} Majority Class OOF Macro F1: {f1:.4f}")
    return f1

print("\n  Subtask 1 baselines …")
maj_t1 = majority_baseline(t1_int, "T1")
lr_t1, lr_t1_report = run_baseline(t1_int, "T1 (TF-IDF+LR)", T1_LABELS)

print("\n  Subtask 2 baselines …")
maj_t2 = majority_baseline(t2_int, "T2")
lr_t2, lr_t2_report = run_baseline(t2_int, "T2 (TF-IDF+LR)", T2_LABELS)

# Build comparison table
comparison = {
    "System": [
        "Majority class",
        "TF-IDF + Logistic Regression",
        "DeBERTa-xlarge (single model, 1 seed)",
        "DeBERTa-v3-large (50 models, OOF)",
        "DeBERTa-xlarge (50 models, OOF)",
        "DeBERTa-xlarge ensemble (eval)",
        "DeBERTa-v3-large ensemble T2 (eval)",
    ],
    "Subtask 1 (Macro F1)": [
        f"{maj_t1:.4f}",
        f"{lr_t1:.4f}",
        "0.3109 ± 0.0114",  # From summary stats
        "0.6669",
        "0.6794",
        "0.76",
        "—",
    ],
    "Subtask 2 (Macro F1)": [
        f"{maj_t2:.4f}",
        f"{lr_t2:.4f}",
        "0.3273 ± 0.0396",  # T2 single model
        "0.3727",
        "—",
        "—",
        "0.50",
    ],
    "Data Split": [
        "OOF", "OOF", "OOF", "OOF", "OOF", "Eval (official)", "Eval (official)"
    ]
}

comp_df = pd.DataFrame(comparison)
comp_df.to_csv(OUT / "table_baseline_comparison.csv", index=False)
print("\nBaseline Comparison Table:")
print(comp_df.to_string(index=False))

# Per-class comparison for LR vs DeBERTa
print("\n  Per-class F1 comparison (T1): LR vs DeBERTa-xlarge (OOF)")
print(f"  {'Class':<22} {'LR':>8} {'DeBERTa-xl':>12}")
for name in T1_LABELS:
    lr_f1  = lr_t1_report[name]["f1-score"]
    deb_f1 = {
        "Clear Reply": 0.6434, "Ambivalent": 0.7665, "Clear Non-Reply": 0.6283
    }.get(name, 0.0)
    print(f"  {name:<22} {lr_f1:>8.4f} {deb_f1:>12.4f}")

# Save all baseline results
baseline_results = {
    "t1_majority_f1": float(maj_t1),
    "t1_tfidf_lr_f1": float(lr_t1),
    "t1_deberta_xl_single_mean_f1": 0.3109,
    "t1_deberta_xl_ensemble_oof_f1": 0.6794,
    "t1_deberta_xl_ensemble_eval_f1": 0.76,
    "t2_majority_f1": float(maj_t2),
    "t2_tfidf_lr_f1": float(lr_t2),
    "t2_deberta_v3_single_mean_f1": 0.3273,
    "t2_deberta_v3_ensemble_oof_f1": 0.3727,
    "t2_deberta_v3_ensemble_eval_f1": 0.50,
    "t1_lr_per_class": {n: lr_t1_report[n] for n in T1_LABELS},
    "t2_lr_per_class": {n: lr_t2_report[n] for n in T2_LABELS},
}
with open(OUT / "baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

print("\n[OK] All baselines and error examples generated!")
print(f"  Outputs in: {OUT.resolve()}")
