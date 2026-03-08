"""
Data loader for SemEval 2026 Task 6 (CLARITY).

Loads the ailsntua/QEvasion dataset from Hugging Face and provides clean interfaces
for accessing train/test splits with normalized labels.
"""

import os
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd


# Authoritative label mappings for CodaBench submission
CLARITY_LABELS = [
    "Clear Reply",
    "Ambivalent", 
    "Clear Non-Reply"
]

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

# Hierarchical mapping: evasion -> clarity
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


def load_clarity_dataset(cache_dir: Optional[str] = None) -> DatasetDict:
    """
    Load the QEvasion dataset from Hugging Face.
    
    Args:
        cache_dir: Optional cache directory for HF datasets.
                   Defaults to HF_DATASETS_CACHE env var or HF default.
    
    Returns:
        DatasetDict with 'train' and 'test' splits.
    """
    if cache_dir is None:
        cache_dir = os.environ.get("HF_DATASETS_CACHE")
    
    dataset = load_dataset("ailsntua/QEvasion", cache_dir=cache_dir)
    return dataset


def normalize_labels(dataset: Dataset) -> Dataset:
    """
    Normalize labels to match CodaBench submission format exactly.
    
    This ensures no whitespace/capitalization issues in submissions.
    
    Args:
        dataset: HF Dataset with 'clarity_label' and 'evasion_label' columns.
    
    Returns:
        Dataset with normalized labels.
    """
    def _normalize(example):
        # Normalize clarity label
        if example["clarity_label"] in CLARITY_LABELS:
            example["clarity_label"] = example["clarity_label"]
        else:
            # Try to fix common issues
            label = example["clarity_label"].strip()
            if label in CLARITY_LABELS:
                example["clarity_label"] = label
            else:
                # Log warning but keep original
                print(f"Warning: Unknown clarity label: {example['clarity_label']}")
        
        # Normalize evasion label
        if example["evasion_label"] in EVASION_LABELS:
            example["evasion_label"] = example["evasion_label"]
        else:
            label = example["evasion_label"].strip()
            if label in EVASION_LABELS:
                example["evasion_label"] = label
            else:
                print(f"Warning: Unknown evasion label: {example['evasion_label']}")
        
        return example
    
    return dataset.map(_normalize)


def add_label_ids(dataset: Dataset) -> Dataset:
    """
    Add integer label IDs for training.
    
    Args:
        dataset: Dataset with string labels.
    
    Returns:
        Dataset with 'clarity_label_id' and 'evasion_label_id' columns.
    """
    clarity_to_id = {label: i for i, label in enumerate(CLARITY_LABELS)}
    evasion_to_id = {label: i for i, label in enumerate(EVASION_LABELS)}
    
    def _add_ids(example):
        example["clarity_label_id"] = clarity_to_id.get(
            example["clarity_label"], -1
        )
        example["evasion_label_id"] = evasion_to_id.get(
            example["evasion_label"], -1
        )
        return example
    
    return dataset.map(_add_ids)


def verify_taxonomy_consistency(dataset: Dataset) -> Dict[str, int]:
    """
    Verify that evasion labels map correctly to clarity labels.
    
    Args:
        dataset: Dataset with both label types.
    
    Returns:
        Dict with counts of inconsistent examples.
    """
    inconsistencies = {
        "total": 0,
        "examples": []
    }
    
    for i, example in enumerate(dataset):
        expected_clarity = EVASION_TO_CLARITY.get(example["evasion_label"])
        actual_clarity = example["clarity_label"]
        
        if expected_clarity != actual_clarity:
            inconsistencies["total"] += 1
            inconsistencies["examples"].append({
                "index": i,
                "evasion_label": example["evasion_label"],
                "expected_clarity": expected_clarity,
                "actual_clarity": actual_clarity
            })
    
    return inconsistencies


def get_label_distribution(dataset: Dataset) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get label distribution for both tasks.
    
    Args:
        dataset: Dataset with labels.
    
    Returns:
        Tuple of (clarity_dist, evasion_dist) as DataFrames.
    """
    df = dataset.to_pandas()
    
    clarity_dist = df["clarity_label"].value_counts().reset_index()
    clarity_dist.columns = ["label", "count"]
    clarity_dist["percentage"] = 100 * clarity_dist["count"] / len(df)
    
    evasion_dist = df["evasion_label"].value_counts().reset_index()
    evasion_dist.columns = ["label", "count"]
    evasion_dist["percentage"] = 100 * evasion_dist["count"] / len(df)
    
    return clarity_dist, evasion_dist


def main():
    """Example usage and verification."""
    print("Loading dataset...")
    dataset = load_clarity_dataset()
    
    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")
    
    print("\n" + "="*60)
    print("TRAIN SET ANALYSIS")
    print("="*60)
    
    # Normalize labels
    train = normalize_labels(dataset["train"])
    train = add_label_ids(train)
    
    # Check taxonomy consistency
    print("\nVerifying taxonomy consistency...")
    inconsistencies = verify_taxonomy_consistency(train)
    if inconsistencies["total"] > 0:
        print(f"⚠️  Found {inconsistencies['total']} inconsistent examples")
        print("First few:", inconsistencies["examples"][:3])
    else:
        print("[OK] All labels are taxonomy-consistent!")
    
    # Label distributions
    print("\nLabel distributions:")
    clarity_dist, evasion_dist = get_label_distribution(train)
    
    print("\nClarity labels:")
    print(clarity_dist.to_string(index=False))
    
    print("\nEvasion labels:")
    print(evasion_dist.to_string(index=False))
    
    # Test set
    print("\n" + "="*60)
    print("TEST SET ANALYSIS")
    print("="*60)
    test = normalize_labels(dataset["test"])
    test = add_label_ids(test)
    
    clarity_dist_test, evasion_dist_test = get_label_distribution(test)
    print("\nClarity labels:")
    print(clarity_dist_test.to_string(index=False))
    
    print("\nEvasion labels:")
    print(evasion_dist_test.to_string(index=False))


if __name__ == "__main__":
    main()
