#!/usr/bin/env python3
"""
Comprehensive Paper Analysis Script
SemEval-2026 Task 6: CLARITY

Generates all figures, tables, and analyses needed for the paper:
  1. Dataset class distribution
  2. Ensemble size ablation (T1 xlarge, T1 v3-large, T2 v3-large)
  3. Per-seed variance analysis
  4. Confusion matrices (T1, T2)
  5. Per-class F1 breakdown
  6. Error analysis (misclassified examples)
  7. Cross-task correlation
  
All outputs -> docs/paper_figures/
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Labels ──────────────────────────────────────────────────────────────────
T1_LABELS    = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
T2_LABELS    = ["Claims ignorance", "Clarification", "Declining to answer",
                "Deflection", "Dodging", "Explicit", "General",
                "Implicit", "Partial/half-answer"]

OUT = Path("docs/paper_figures")
OUT.mkdir(parents=True, exist_ok=True)

# Pretty plot defaults
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.dpi": 300,
})
PALETTE = sns.color_palette("muted")

# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Loading OOF data …")
    t1x = np.load("artifacts/oof/task1_10seed_oof.npz")
    t1v = np.load("artifacts/oof/task1_v3large_oof.npz")
    t2v = np.load("artifacts/oof/task2_v3large_oof.npz")

    # T1 xlarge: logits (3448, 3, 50), labels (3448,)
    t1x_logits = t1x["logits"]          # (3448, 3, 50)
    t1x_labels = t1x["labels"]          # (3448,)
    t1x_mask   = t1x["val_mask"]        # (3448,)  True where sample was in val fold

    # T1 v3-large: logits (3448, 3, 50), ground_truth (3448,), mask (3448, 50)
    t1v_logits = t1v["logits"]          # (3448, 3, 50)
    t1v_labels = t1v["ground_truth"]    # (3448,)
    t1v_mask   = t1v["mask"]            # (3448, 50) — per-model val mask

    # T2 v3-large
    t2v_logits = t2v["logits"]          # (3448, 9, 50)
    t2v_labels = t2v["ground_truth"]    # (3448,)
    t2v_mask   = t2v["mask"]            # (3448, 50)

    # Load training CSV for dataset stats
    train_df = None
    for p in ["data/train.csv", "data/clarity_dataset.csv"]:
        if os.path.exists(p):
            train_df = pd.read_csv(p)
            break

    return {
        "t1x_logits": t1x_logits, "t1x_labels": t1x_labels, "t1x_mask": t1x_mask,
        "t1v_logits": t1v_logits, "t1v_labels": t1v_labels, "t1v_mask": t1v_mask,
        "t2v_logits": t2v_logits, "t2v_labels": t2v_labels, "t2v_mask": t2v_mask,
        "train_df": train_df,
    }


# ════════════════════════════════════════════════════════════════════════════
# HELPER: macro F1 on OOF predictions
# ════════════════════════════════════════════════════════════════════════════

def oof_macro_f1_xlarge(logits_3d, labels, mask_1d, n_models):
    """
    logits_3d : (N, C, M)  — all 50 models' logits
    mask_1d   : (N,)       — True when sample was NOT in training for that fold
    n_models  : how many models to average
    """
    avg = logits_3d[:, :, :n_models].mean(axis=2)   # (N, C)
    preds = avg.argmax(axis=1)                        # (N,)
    # Only evaluate on samples that were in validation at some point
    idx = np.where(mask_1d)[0]
    return f1_score(labels[idx], preds[idx], average="macro")


def oof_macro_f1_v3(logits_3d, labels, mask_2d, n_models):
    """
    mask_2d : (N, M) — True when sample i was in val for model m
    Average only models 0..n_models-1; for each sample use models where it was in val.
    """
    # For a fair comparison, use ensemble_logits when averaging subset
    # Sum logits only from models where sample was in val
    sub_logits = logits_3d[:, :, :n_models]   # (N, C, n_models)
    sub_mask   = mask_2d[:, :n_models]         # (N, n_models)
    # Weighted sum then divide by count
    weighted = (sub_logits * sub_mask[:, np.newaxis, :]).sum(axis=2)  # (N, C)
    counts   = sub_mask.sum(axis=1, keepdims=True).clip(min=1)         # (N, 1)
    avg      = weighted / counts                                         # (N, C)
    preds    = avg.argmax(axis=1)
    # Evaluate on samples that appear in val for at least one model in subset
    idx = np.where(sub_mask.any(axis=1))[0]
    return f1_score(labels[idx], preds[idx], average="macro")


# ════════════════════════════════════════════════════════════════════════════
# 1. DATASET CLASS DISTRIBUTION
# ════════════════════════════════════════════════════════════════════════════

def plot_dataset_distribution(data):
    print("\n[1] Dataset class distribution …")

    # Use OOF ground truth as proxy (= training set)
    t1_labels = data["t1x_labels"]
    t2_labels = data["t2v_labels"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # — Subtask 1 —
    ax = axes[0]
    counts = [np.sum(t1_labels == i) for i in range(3)]
    total  = len(t1_labels)
    bars = ax.bar(T1_LABELS, counts, color=PALETTE[:3], edgecolor="white", linewidth=0.8)
    ax.set_title("Subtask 1 — Clarity Level Distribution\n(Training set, N=3,448)")
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.18)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f"{c}\n({c/total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", labelrotation=10)

    # — Subtask 2 —
    ax = axes[1]
    counts2 = [np.sum(t2_labels == i) for i in range(9)]
    total2  = len(t2_labels)
    sorted_idx = np.argsort(counts2)[::-1]
    sorted_counts = [counts2[i] for i in sorted_idx]
    sorted_names  = [T2_LABELS[i] for i in sorted_idx]
    bars2 = ax.bar(range(9), sorted_counts, color=PALETTE, edgecolor="white", linewidth=0.8)
    ax.set_title("Subtask 2 — Evasion Type Distribution\n(Training set, N=3,448)")
    ax.set_ylabel("Count")
    ax.set_xticks(range(9))
    ax.set_xticklabels(sorted_names, rotation=35, ha="right", fontsize=9)
    ax.set_ylim(0, max(sorted_counts) * 1.15)
    for bar, c in zip(bars2, sorted_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f"{c}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "fig1_dataset_distribution.pdf")
    plt.savefig(OUT / "fig1_dataset_distribution.png")
    plt.close()

    # Print table
    print("\nTable: Subtask 1 class distribution")
    for i, name in enumerate(T1_LABELS):
        print(f"  {name}: {counts[i]} ({counts[i]/total:.1%})")
    print("\nTable: Subtask 2 class distribution (sorted)")
    for i in sorted_idx:
        print(f"  {T2_LABELS[i]}: {counts2[i]} ({counts2[i]/total2:.1%})")


# ════════════════════════════════════════════════════════════════════════════
# 2. ENSEMBLE SIZE ABLATION
# ════════════════════════════════════════════════════════════════════════════

def ensemble_size_ablation(data):
    print("\n[2] Ensemble size ablation …")

    sizes = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50]

    f1_t1x, f1_t1v, f1_t2v = [], [], []

    for n in sizes:
        f1_t1x.append(oof_macro_f1_xlarge(
            data["t1x_logits"], data["t1x_labels"], data["t1x_mask"], n))
        f1_t1v.append(oof_macro_f1_v3(
            data["t1v_logits"], data["t1v_labels"], data["t1v_mask"], n))
        f1_t2v.append(oof_macro_f1_v3(
            data["t2v_logits"], data["t2v_labels"], data["t2v_mask"], n))

        print(f"  n={n:3d}: T1-xl={f1_t1x[-1]:.4f} | T1-v3={f1_t1v[-1]:.4f} | T2-v3={f1_t2v[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subtask 1
    ax = axes[0]
    ax.plot(sizes, f1_t1x, "o-", color=PALETTE[0], linewidth=2, markersize=6,
            label="DeBERTa-xlarge (Subtask 1)")
    ax.plot(sizes, f1_t1v, "s--", color=PALETTE[1], linewidth=2, markersize=6,
            label="DeBERTa-v3-large (Subtask 1)")
    ax.set_xlabel("Number of models in ensemble")
    ax.set_ylabel("OOF Macro F1")
    ax.set_title("Subtask 1: Effect of Ensemble Size")
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0.60, 0.72)

    # Subtask 2
    ax = axes[1]
    ax.plot(sizes, f1_t2v, "^-", color=PALETTE[2], linewidth=2, markersize=6,
            label="DeBERTa-v3-large (Subtask 2)")
    ax.set_xlabel("Number of models in ensemble")
    ax.set_ylabel("OOF Macro F1")
    ax.set_title("Subtask 2: Effect of Ensemble Size")
    ax.legend()
    ax.grid(True, alpha=0.35)
    ax.set_ylim(0.30, 0.42)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_ensemble_size_ablation.pdf")
    plt.savefig(OUT / "fig2_ensemble_size_ablation.png")
    plt.close()

    # Save table
    ablation_df = pd.DataFrame({
        "n_models": sizes,
        "T1_xlarge_OOF_F1": f1_t1x,
        "T1_v3large_OOF_F1": f1_t1v,
        "T2_v3large_OOF_F1": f1_t2v,
    })
    ablation_df.to_csv(OUT / "table_ensemble_ablation.csv", index=False)
    print(ablation_df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# 3. PER-SEED VARIANCE ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def seed_variance_analysis(data):
    print("\n[3] Per-seed variance analysis …")

    results = {"model": [], "seed": [], "f1": []}

    # T1 xlarge: each model's OOF F1
    mask1d = data["t1x_mask"]
    idx    = np.where(mask1d)[0]
    for m in range(50):
        preds = data["t1x_logits"][idx, :, m].argmax(axis=1)
        f1 = f1_score(data["t1x_labels"][idx], preds, average="macro")
        results["model"].append("DeBERTa-xlarge (T1)")
        results["seed"].append(m)
        results["f1"].append(f1)

    # T1 v3-large: per-model OOF F1 (only on its own validation samples)
    for m in range(50):
        m_idx = np.where(data["t1v_mask"][:, m])[0]
        if len(m_idx) == 0:
            continue
        preds = data["t1v_logits"][m_idx, :, m].argmax(axis=1)
        f1 = f1_score(data["t1v_labels"][m_idx], preds, average="macro")
        results["model"].append("DeBERTa-v3-large (T1)")
        results["seed"].append(m)
        results["f1"].append(f1)

    # T2 v3-large
    for m in range(50):
        m_idx = np.where(data["t2v_mask"][:, m])[0]
        if len(m_idx) == 0:
            continue
        preds = data["t2v_logits"][m_idx, :, m].argmax(axis=1)
        f1 = f1_score(data["t2v_labels"][m_idx], preds, average="macro")
        results["model"].append("DeBERTa-v3-large (T2)")
        results["seed"].append(m)
        results["f1"].append(f1)

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = ["DeBERTa-xlarge (T1)", "DeBERTa-v3-large (T1)", "DeBERTa-v3-large (T2)"]
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    for i, (model, color) in enumerate(zip(order, colors)):
        vals = df[df["model"] == model]["f1"].values
        parts = ax.violinplot([vals], positions=[i], widths=0.55,
                               showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cbars"].set_color(color)
        parts["cmins"].set_color(color)
        parts["cmaxes"].set_color(color)
        ax.scatter([i]*len(vals), vals, color=color, s=18, alpha=0.55, zorder=3)
        mu, std = vals.mean(), vals.std()
        print(f"  {model}: mean={mu:.4f} ± {std:.4f} (min={vals.min():.4f}, max={vals.max():.4f})")

    ax.set_xticks(range(3))
    ax.set_xticklabels(order, rotation=10)
    ax.set_ylabel("Single-model OOF Macro F1")
    ax.set_title("Per-model F1 Variance Across Seeds (N=50 per model)")
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT / "fig3_seed_variance.pdf")
    plt.savefig(OUT / "fig3_seed_variance.png")
    plt.close()

    df.to_csv(OUT / "table_seed_variance.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════
# 4. CONFUSION MATRICES
# ════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(data):
    print("\n[4] Confusion matrices …")

    # ── Subtask 1: xlarge ensemble ──
    mask1d   = data["t1x_mask"]
    idx      = np.where(mask1d)[0]
    avg_logits = data["t1x_logits"][idx].mean(axis=2)  # (N_val, 3)
    t1_preds   = avg_logits.argmax(axis=1)
    t1_true    = data["t1x_labels"][idx]

    cm1 = confusion_matrix(t1_true, t1_preds)
    cm1_norm = cm1.astype(float) / cm1.sum(axis=1, keepdims=True)

    # ── Subtask 2: v3-large ensemble ──
    weighted2 = (data["t2v_logits"] * data["t2v_mask"][:, np.newaxis, :]).sum(axis=2)
    counts2   = data["t2v_mask"].sum(axis=1, keepdims=True).clip(min=1)
    avg2      = weighted2 / counts2
    idx2      = np.where(data["t2v_mask"].any(axis=1))[0]
    t2_preds  = avg2[idx2].argmax(axis=1)
    t2_true   = data["t2v_labels"][idx2]

    cm2 = confusion_matrix(t2_true, t2_preds)
    cm2_norm = cm2.astype(float) / cm2.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # T1
    sns.heatmap(cm1_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=T1_LABELS, yticklabels=T1_LABELS,
                ax=axes[0], linewidths=0.5, cbar_kws={"shrink": 0.8})
    # Overlay counts
    for i in range(3):
        for j in range(3):
            axes[0].text(j+0.5, i+0.72, f"n={cm1[i,j]}",
                         ha="center", va="center", fontsize=8, color="gray")
    axes[0].set_title("Subtask 1 — Confusion Matrix (OOF)\nDeBERTa-xlarge Ensemble (50 models)")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # T2 — short labels
    t2_short = ["Clm.Ign.", "Clar.", "Decl.", "Deflect.", "Dodging",
                "Explicit", "General", "Implicit", "Partial"]
    sns.heatmap(cm2_norm, annot=True, fmt=".2f", cmap="Oranges",
                xticklabels=t2_short, yticklabels=t2_short,
                ax=axes[1], linewidths=0.5, cbar_kws={"shrink": 0.8})
    axes[1].set_title("Subtask 2 — Confusion Matrix (OOF)\nDeBERTa-v3-large Ensemble (50 models)")
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(OUT / "fig4_confusion_matrices.pdf")
    plt.savefig(OUT / "fig4_confusion_matrices.png")
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# 5. PER-CLASS F1 BREAKDOWN
# ════════════════════════════════════════════════════════════════════════════

def per_class_f1(data):
    print("\n[5] Per-class F1 breakdown …")

    # T1 xlarge
    mask1d = data["t1x_mask"]
    idx    = np.where(mask1d)[0]
    t1_avg = data["t1x_logits"][idx].mean(axis=2)
    t1_pred = t1_avg.argmax(axis=1)
    t1_true = data["t1x_labels"][idx]
    t1_report = classification_report(t1_true, t1_pred,
                                       target_names=T1_LABELS, output_dict=True)

    # T2 v3-large
    weighted2 = (data["t2v_logits"] * data["t2v_mask"][:, np.newaxis, :]).sum(axis=2)
    counts2   = data["t2v_mask"].sum(axis=1, keepdims=True).clip(min=1)
    avg2      = weighted2 / counts2
    idx2      = np.where(data["t2v_mask"].any(axis=1))[0]
    t2_pred   = avg2[idx2].argmax(axis=1)
    t2_true   = data["t2v_labels"][idx2]
    t2_report = classification_report(t2_true, t2_pred,
                                       target_names=T2_LABELS, output_dict=True)

    # Print LaTeX-friendly tables
    print("\n--- Subtask 1 per-class results (OOF) ---")
    print(f"{'Class':<22} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    for name in T1_LABELS:
        r = t1_report[name]
        print(f"{name:<22} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1-score']:>8.4f} {r['support']:>9}")
    print(f"{'Macro avg':<22} {t1_report['macro avg']['precision']:>10.4f} "
          f"{t1_report['macro avg']['recall']:>8.4f} "
          f"{t1_report['macro avg']['f1-score']:>8.4f}")

    print("\n--- Subtask 2 per-class results (OOF) ---")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    for name in T2_LABELS:
        r = t2_report[name]
        print(f"{name:<25} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1-score']:>8.4f} {r['support']:>9}")
    print(f"{'Macro avg':<25} {t2_report['macro avg']['precision']:>10.4f} "
          f"{t2_report['macro avg']['recall']:>8.4f} "
          f"{t2_report['macro avg']['f1-score']:>8.4f}")

    # Bar chart: per-class F1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # T1
    ax = axes[0]
    t1_f1s = [t1_report[n]["f1-score"] for n in T1_LABELS]
    bars = ax.bar(T1_LABELS, t1_f1s, color=PALETTE[:3], edgecolor="white")
    ax.axhline(t1_report["macro avg"]["f1-score"], color="red", linestyle="--",
               linewidth=1.5, label=f"Macro avg = {t1_report['macro avg']['f1-score']:.3f}")
    ax.set_title("Subtask 1: Per-class F1 Score (OOF)")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    for bar, v in zip(bars, t1_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    # T2
    ax = axes[1]
    t2_f1s = [t2_report[n]["f1-score"] for n in T2_LABELS]
    bars2 = ax.bar(range(9), t2_f1s, color=PALETTE, edgecolor="white")
    ax.axhline(t2_report["macro avg"]["f1-score"], color="red", linestyle="--",
               linewidth=1.5, label=f"Macro avg = {t2_report['macro avg']['f1-score']:.3f}")
    ax.set_xticks(range(9))
    ax.set_xticklabels(T2_LABELS, rotation=40, ha="right", fontsize=8)
    ax.set_title("Subtask 2: Per-class F1 Score (OOF)")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.legend()
    for bar, v in zip(bars2, t2_f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "fig5_per_class_f1.pdf")
    plt.savefig(OUT / "fig5_per_class_f1.png")
    plt.close()

    # Save tables
    t1_rows = [{
        "Class": n, "Precision": t1_report[n]["precision"],
        "Recall": t1_report[n]["recall"], "F1": t1_report[n]["f1-score"],
        "Support": int(t1_report[n]["support"])
    } for n in T1_LABELS]
    pd.DataFrame(t1_rows).to_csv(OUT / "table_t1_per_class_f1.csv", index=False)

    t2_rows = [{
        "Class": n, "Precision": t2_report[n]["precision"],
        "Recall": t2_report[n]["recall"], "F1": t2_report[n]["f1-score"],
        "Support": int(t2_report[n]["support"])
    } for n in T2_LABELS]
    pd.DataFrame(t2_rows).to_csv(OUT / "table_t2_per_class_f1.csv", index=False)

    return t1_report, t2_report


# ════════════════════════════════════════════════════════════════════════════
# 6. ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

def error_analysis(data):
    print("\n[6] Error analysis …")

    # Load training data to get text
    train_path = None
    for p in ["data/train.csv", "data/clarity_dataset.csv", "data/qevasion_train.csv"]:
        if os.path.exists(p):
            train_path = p
            break

    # T1 errors
    mask1d = data["t1x_mask"]
    idx    = np.where(mask1d)[0]
    t1_avg = data["t1x_logits"][idx].mean(axis=2)
    t1_pred = t1_avg.argmax(axis=1)
    t1_true = data["t1x_labels"][idx]
    t1_conf = t1_avg.max(axis=1) / t1_avg.sum(axis=1)  # relative confidence

    t1_errors_idx = np.where(t1_pred != t1_true)[0]
    # Top confused pairs
    from collections import Counter
    pairs = [(T1_LABELS[t1_true[i]], T1_LABELS[t1_pred[i]]) for i in t1_errors_idx]
    print(f"\n  T1 total errors: {len(t1_errors_idx)} / {len(idx)} ({len(t1_errors_idx)/len(idx):.1%})")
    print("  Most confused pairs (true -> predicted):")
    for (t, p), c in Counter(pairs).most_common(6):
        print(f"    {t} -> {p}: {c}")

    # T2 errors
    weighted2 = (data["t2v_logits"] * data["t2v_mask"][:, np.newaxis, :]).sum(axis=2)
    counts2   = data["t2v_mask"].sum(axis=1, keepdims=True).clip(min=1)
    avg2      = weighted2 / counts2
    idx2      = np.where(data["t2v_mask"].any(axis=1))[0]
    t2_pred   = avg2[idx2].argmax(axis=1)
    t2_true   = data["t2v_labels"][idx2]

    t2_errors_idx = np.where(t2_pred != t2_true)[0]
    pairs2 = [(T2_LABELS[t2_true[i]], T2_LABELS[t2_pred[i]]) for i in t2_errors_idx]
    print(f"\n  T2 total errors: {len(t2_errors_idx)} / {len(idx2)} ({len(t2_errors_idx)/len(idx2):.1%})")
    print("  Most confused pairs (true -> predicted):")
    for (t, p), c in Counter(pairs2).most_common(8):
        print(f"    {t} -> {p}: {c}")

    # Error breakdown report
    error_data = {
        "t1_error_rate": len(t1_errors_idx)/len(idx),
        "t2_error_rate": len(t2_errors_idx)/len(idx2),
        "t1_top_confusions": [{"true": t, "pred": p, "count": c}
                               for (t, p), c in Counter(pairs).most_common(6)],
        "t2_top_confusions": [{"true": t, "pred": p, "count": c}
                               for (t, p), c in Counter(pairs2).most_common(8)],
    }

    if train_path:
        df = pd.read_csv(train_path)
        print(f"\n  Loaded training data: {len(df)} rows, columns: {list(df.columns)}")

        # Try to extract misclassified examples
        try:
            # Map OOF indices back to training df
            # This works if df is in same order as OOF
            q_col   = [c for c in df.columns if "question" in c.lower()][0]
            ans_col = [c for c in df.columns if "answer" in c.lower()][0]

            # T1 errors — pick high-confidence mistakes (most interesting)
            orig_indices = idx[t1_errors_idx]
            conf_scores  = t1_conf[t1_errors_idx]
            # Sort by confidence (most confidently wrong)
            sort_order = np.argsort(conf_scores)[::-1][:20]
            examples = []
            for si in sort_order[:10]:
                oi = orig_indices[si]
                if oi < len(df):
                    examples.append({
                        "orig_idx": int(oi),
                        "question": df.iloc[oi][q_col][:200],
                        "answer": df.iloc[oi][ans_col][:300],
                        "true_label": T1_LABELS[t1_true[t1_errors_idx[si]]],
                        "pred_label": T1_LABELS[t1_pred[t1_errors_idx[si]]],
                        "confidence": float(conf_scores[si]),
                    })
            error_data["t1_examples"] = examples
            print(f"\n  Extracted {len(examples)} T1 error examples for qualitative analysis")
            for ex in examples[:5]:
                print(f"\n  --- idx {ex['orig_idx']} ---")
                print(f"  TRUE: {ex['true_label']}  PRED: {ex['pred_label']}  CONF: {ex['confidence']:.3f}")
                print(f"  Q: {ex['question'][:120]}")
                print(f"  A: {ex['answer'][:180]}")
        except Exception as e:
            print(f"  (Could not extract text examples: {e})")
    else:
        print("\n  ⚠  Training CSV not found — skipping text examples.")
        print("  Note: Run after placing data CSV in data/ folder.")

    with open(OUT / "error_analysis.json", "w") as f:
        json.dump(error_data, f, indent=2, ensure_ascii=False)


# ════════════════════════════════════════════════════════════════════════════
# 7. OOF vs EVAL OPTIMIZATION TABLE
# ════════════════════════════════════════════════════════════════════════════

def optimization_paradox_table():
    print("\n[7] Optimization paradox table …")

    rows = [
        {"Optimization":      "Single-fold baseline",
         "OOF Δ (T1)": "+0.000", "Eval Δ (T1)": "0.54",
         "OOF Δ (T2)": "+0.000", "Eval Δ (T2)": "0.50"},
        {"Optimization":      "Multi-seed ensemble (50 models)",
         "OOF Δ (T1)": "+0.012", "Eval Δ (T1)": "+0.22 (->0.76)",
         "OOF Δ (T2)": "+0.010", "Eval Δ (T2)": "0.00 (->0.50)"},
        {"Optimization":      "Optuna weight optimization",
         "OOF Δ (T1)": "+0.003", "Eval Δ (T1)": "−0.02 (->0.74)",
         "OOF Δ (T2)": "+0.003", "Eval Δ (T2)": "—"},
        {"Optimization":      "Class threshold calibration",
         "OOF Δ (T1)": "+0.007", "Eval Δ (T1)": "—",
         "OOF Δ (T2)": "+0.007", "Eval Δ (T2)": "—"},
        {"Optimization":      "Learned hierarchical masking",
         "OOF Δ (T1)": "—",     "Eval Δ (T1)": "—",
         "OOF Δ (T2)": "+0.026", "Eval Δ (T2)": "−0.10 (->0.40)"},
        {"Optimization":      "Mega-ensemble (233+ models)",
         "OOF Δ (T1)": "+0.001", "Eval Δ (T1)": "0.00",
         "OOF Δ (T2)": "+0.001", "Eval Δ (T2)": "—"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "table_optimization_paradox.csv", index=False)
    print(df.to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY STATS JSON (for paper writing)
# ════════════════════════════════════════════════════════════════════════════

def generate_summary_stats(data):
    print("\n[8] Summary statistics …")

    mask1d = data["t1x_mask"]
    idx    = np.where(mask1d)[0]
    t1_avg  = data["t1x_logits"][idx].mean(axis=2)
    t1_pred = t1_avg.argmax(axis=1)
    t1_true = data["t1x_labels"][idx]
    t1_macro = f1_score(t1_true, t1_pred, average="macro")
    t1v_macro = float(np.load("artifacts/oof/task1_v3large_oof.npz")["ensemble_f1"])
    t2_macro  = float(np.load("artifacts/oof/task2_v3large_oof.npz")["ensemble_f1"])

    # Per-seed stats for xlarge
    per_seed_f1 = []
    for m in range(50):
        p = data["t1x_logits"][idx, :, m].argmax(axis=1)
        per_seed_f1.append(f1_score(data["t1x_labels"][idx], p, average="macro"))
    per_seed_f1 = np.array(per_seed_f1)

    stats = {
        "t1_xlarge_oof_ensemble_f1": float(t1_macro),
        "t1_v3large_oof_ensemble_f1": float(t1v_macro),
        "t2_v3large_oof_ensemble_f1": float(t2_macro),
        "t1_xlarge_single_model_mean_f1": float(per_seed_f1.mean()),
        "t1_xlarge_single_model_std_f1":  float(per_seed_f1.std()),
        "t1_xlarge_single_model_min_f1":  float(per_seed_f1.min()),
        "t1_xlarge_single_model_max_f1":  float(per_seed_f1.max()),
        "eval_scores": {
            "t1_best": 0.76, "t1_baseline": 0.54, "t1_overoptimized": 0.74,
            "t2_best": 0.50, "t2_baseline": 0.50, "t2_overoptimized": 0.40,
        },
        "training_samples": 3448,
        "eval_samples": 237,
        "n_models_t1_xlarge": 50,
        "n_models_t1_v3large": 50,
        "n_models_t2_v3large": 50,
    }

    with open(OUT / "summary_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return stats


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("="*65)
    print("PAPER ANALYSIS: SemEval-2026 Task 6 CLARITY")
    print("="*65)

    data = load_data()

    plot_dataset_distribution(data)
    ensemble_size_ablation(data)
    seed_variance_analysis(data)
    plot_confusion_matrices(data)
    t1_report, t2_report = per_class_f1(data)
    error_analysis(data)
    optimization_paradox_table()
    stats = generate_summary_stats(data)

    print("\n" + "="*65)
    print(f"[OK] All analyses complete. Outputs in: {OUT.resolve()}")
    print("="*65)
    print("\nGenerated files:")
    for f in sorted(OUT.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
