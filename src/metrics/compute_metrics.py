"""
Metrics computation for CLARITY task.

Implements:
- Macro F1 (primary metric for both subtasks)
- Per-class F1
- Confusion matrix
- Taxonomy consistency check
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from typing import Dict, List, Tuple, Optional
import pandas as pd


# Label mappings from load_dataset.py
EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification"
]

CLARITY_LABELS = [
    "Clear Reply",
    "Ambivalent",
    "Clear Non-Reply"
]

EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro F1 score.
    
    Args:
        y_true: True labels (indices)
        y_pred: Predicted labels (indices)
    
    Returns:
        Macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, F1.
    
    Args:
        y_true: True labels (indices)
        y_pred: Predicted labels (indices)
        label_names: List of label names
    
    Returns:
        Dict mapping label_name -> {precision, recall, f1, support}
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    metrics = {}
    for i, label in enumerate(label_names):
        metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred)


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str]
) -> str:
    """
    Get sklearn classification report as string.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Label names
    
    Returns:
        Classification report string
    """
    return classification_report(
        y_true, y_pred,
        target_names=label_names,
        zero_division=0
    )


def derive_clarity_from_evasion(
    evasion_preds: np.ndarray,
    return_indices: bool = True
) -> np.ndarray:
    """
    Derive clarity predictions from evasion predictions via taxonomy mapping.
    
    Args:
        evasion_preds: Evasion label indices
        return_indices: If True, return clarity indices; if False, return label strings
    
    Returns:
        Clarity predictions (indices or strings)
    """
    clarity_preds = []
    
    for evasion_idx in evasion_preds:
        evasion_label = EVASION_LABELS[evasion_idx]
        clarity_label = EVASION_TO_CLARITY[evasion_label]
        
        if return_indices:
            clarity_idx = CLARITY_LABELS.index(clarity_label)
            clarity_preds.append(clarity_idx)
        else:
            clarity_preds.append(clarity_label)
    
    return np.array(clarity_preds)


def check_taxonomy_consistency(
    evasion_preds: np.ndarray,
    clarity_preds: np.ndarray
) -> Tuple[bool, int]:
    """
    Check if evasion and clarity predictions are taxonomy-consistent.
    
    Args:
        evasion_preds: Evasion label indices
        clarity_preds: Clarity label indices
    
    Returns:
        Tuple of (all_consistent, num_inconsistent)
    """
    expected_clarity = derive_clarity_from_evasion(evasion_preds, return_indices=True)
    inconsistent = np.sum(expected_clarity != clarity_preds)
    
    return inconsistent == 0, int(inconsistent)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
    prefix: str = ""
) -> Dict[str, any]:
    """
    Compute all metrics for a prediction set.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Label names
        prefix: Prefix for metric names (e.g., 'val_')
    
    Returns:
        Dict of all metrics
    """
    metrics = {}
    
    # Macro F1
    metrics[f"{prefix}macro_f1"] = compute_macro_f1(y_true, y_pred)
    
    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_pred, label_names)
    metrics[f"{prefix}per_class"] = per_class
    
    # Overall accuracy
    metrics[f"{prefix}accuracy"] = float(np.mean(y_true == y_pred))
    
    # Confusion matrix
    cm = compute_confusion_matrix(y_true, y_pred)
    metrics[f"{prefix}confusion_matrix"] = cm
    
    return metrics


def print_metrics_summary(metrics: Dict, label_names: List[str]):
    """
    Print a formatted summary of metrics.
    
    Args:
        metrics: Metrics dict from compute_all_metrics
        label_names: Label names
    """
    # Find prefix
    prefix = ""
    for key in metrics.keys():
        if "macro_f1" in key:
            prefix = key.replace("macro_f1", "")
            break
    
    print(f"\n{'='*60}")
    print(f"METRICS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nMacro F1: {metrics[f'{prefix}macro_f1']:.4f}")
    print(f"Accuracy: {metrics[f'{prefix}accuracy']:.4f}")
    
    print(f"\nPer-class F1 scores:")
    per_class = metrics[f'{prefix}per_class']
    
    # Create DataFrame for nice formatting
    df = pd.DataFrame(per_class).T
    df = df.sort_values('f1', ascending=False)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    n = 100
    
    # Simulate evasion predictions
    y_true = np.random.randint(0, len(EVASION_LABELS), n)
    y_pred = y_true.copy()
    # Add some errors
    errors = np.random.choice(n, size=10, replace=False)
    y_pred[errors] = np.random.randint(0, len(EVASION_LABELS), len(errors))
    
    print("Testing metrics on evasion task...")
    metrics = compute_all_metrics(y_true, y_pred, EVASION_LABELS, prefix="test_")
    print_metrics_summary(metrics, EVASION_LABELS)
    
    # Test taxonomy derivation
    print("\nTesting clarity derivation from evasion...")
    clarity_derived = derive_clarity_from_evasion(y_pred)
    print(f"Derived {len(clarity_derived)} clarity predictions")
    print(f"Clarity distribution: {np.bincount(clarity_derived)}")
