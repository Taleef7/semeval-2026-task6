"""
Local test set scorer using HF dataset labels.

NOTE: This uses SINGLE-LABEL ground truth from HF dataset, while CodaBench
evaluation uses MULTI-ANNOTATOR acceptance. Therefore:
- This scorer provides a LOWER BOUND estimate
- Actual CodaBench scores will be higher (due to lenient multi-annotator scoring)
- Use this for relative comparison between models, not absolute score prediction
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from datasets import load_dataset
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.load_dataset import EVASION_LABELS, CLARITY_LABELS, EVASION_TO_CLARITY
from src.models.encoder_classifier import EvasionClassifier
from src.data.preprocess import prepare_model_inputs


def load_test_data(cache_dir: Optional[str] = None):
    """Load test split from HF dataset.
    
    NOTE: Test split labels are empty in HF dataset (as expected).
    This function is kept for future use if/when ground truth becomes available.
    For now, we'll generate predictions without computing metrics against ground truth.
    """
    dataset = load_dataset("ailsntua/QEvasion", cache_dir=cache_dir)
    test_data = dataset["test"]
    
    # Test labels are empty - we can't compute metrics
    # Return None for labels
    return test_data, None, None


def generate_predictions(
    checkpoint_path: str,
    test_data,
    model_name: str = "microsoft/deberta-v3-base",
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda"
):
    """Generate predictions on test set."""
    
    # Load model
    model = EvasionClassifier(
        model_name=model_name,
        num_labels=len(EVASION_LABELS),
        dropout=0.1,
        pooling="cls"
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Prepare inputs
    inputs = prepare_model_inputs(
        test_data,
        tokenizer,
        max_length=max_length
    )
    
    # Generate predictions in batches
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_data), batch_size):
            batch_inputs = {
                "input_ids": torch.tensor(inputs["input_ids"][i:i+batch_size]).to(device),
                "attention_mask": torch.tensor(inputs["attention_mask"][i:i+batch_size]).to(device),
            }
            
            if "token_type_ids" in inputs:
                batch_inputs["token_type_ids"] = torch.tensor(
                    inputs["token_type_ids"][i:i+batch_size]
                ).to(device)
            
            outputs = model(**batch_inputs)
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            all_predictions.extend(preds)
    
    return np.array(all_predictions)


def compute_task2_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Compute Task 2 (9-way evasion) metrics."""
    
    # Macro-F1 (primary metric)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(EVASION_LABELS))))
    support = cm.sum(axis=1)
    
    # Per-class results
    per_class = {}
    for i, label in enumerate(EVASION_LABELS):
        per_class[label] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i])
        }
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def derive_clarity_predictions(evasion_preds: List[int]) -> List[int]:
    """Derive Task 1 (clarity) predictions from Task 2 (evasion) via taxonomy."""
    clarity_preds = []
    
    for evasion_id in evasion_preds:
        evasion_label = EVASION_LABELS[evasion_id]
        clarity_label = EVASION_TO_CLARITY[evasion_label]
        clarity_id = CLARITY_LABELS.index(clarity_label)
        clarity_preds.append(clarity_id)
    
    return clarity_preds


def compute_task1_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Compute Task 1 (3-way clarity) metrics."""
    
    # Macro-F1 (primary metric)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLARITY_LABELS))))
    support = cm.sum(axis=1)
    
    # Per-class results
    per_class = {}
    for i, label in enumerate(CLARITY_LABELS):
        per_class[label] = {
            "precision": float(precision_per_class[i]),
            "recall": float(recall_per_class[i]),
            "f1": float(f1_per_class[i]),
            "support": int(support[i])
        }
    
    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local test set scorer (single-label ground truth from HF dataset)"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSON file for results")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base",
                       help="Model name/path")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="HF datasets cache directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("LOCAL TEST SET PREDICTION GENERATOR")
    print("=" * 80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Output: {args.output}")
    print(f"\n⚠️  NOTE: Test set labels are empty in HF dataset (as expected).")
    print("    This script generates predictions only - no metrics computed.")
    print("    To evaluate: submit predictions to CodaBench.\n")
    
    # Load test data
    print("Loading test data from HF dataset...")
    cache_dir = args.cache_dir or os.environ.get("HF_DATASETS_CACHE")
    test_data, evasion_true, clarity_true = load_test_data(cache_dir)
    print(f"[OK] Loaded {len(test_data)} test instances")
    
    # Generate predictions
    print(f"\nGenerating predictions with {args.model_name}...")
    evasion_preds = generate_predictions(
        checkpoint_path=args.checkpoint,
        test_data=test_data,
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )
    print(f"[OK] Generated {len(evasion_preds)} predictions")
    
    # Derive clarity predictions
    clarity_preds = derive_clarity_predictions(evasion_preds.tolist())
    
    # Convert to label strings
    evasion_labels = [EVASION_LABELS[pred] for pred in evasion_preds]
    clarity_labels = [CLARITY_LABELS[pred] for pred in clarity_preds]
    
    # Compute prediction distributions
    from collections import Counter
    evasion_dist = Counter(evasion_labels)
    clarity_dist = Counter(clarity_labels)
    
    # Package results
    results = {
        "checkpoint": args.checkpoint,
        "test_instances": len(test_data),
        "task1_predictions": clarity_labels,
        "task1_distribution": dict(clarity_dist),
        "task2_predictions": evasion_labels,
        "task2_distribution": dict(evasion_dist),
        "note": "No ground truth available for test set - submit to CodaBench for evaluation"
    }
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"\nTask 1 (3-way Clarity) Distribution:")
    for label, count in sorted(clarity_dist.items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count:3d} ({count/len(test_data)*100:5.1f}%)")
    
    print(f"\nTask 2 (9-way Evasion) Distribution:")
    for label, count in sorted(evasion_dist.items(), key=lambda x: -x[1]):
        print(f"  {label:20s}: {count:3d} ({count/len(test_data)*100:5.1f}%)")
    
    print(f"\n[OK] Results saved to: {args.output}")
    print("\n⚠️  To evaluate performance: generate submission and upload to CodaBench")
    print("=" * 80)


if __name__ == "__main__":
    main()
