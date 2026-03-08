"""
Validation split generation for CLARITY task.

Implements GroupKFold by URL to prevent data leakage from shared interview_answer text.
"""

import os
import json
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from datasets import Dataset


def create_grouped_splits(
    dataset: Dataset,
    n_splits: int = 5,
    group_by: str = "url",
    random_state: int = 42
) -> List[Dict]:
    """
    Create GroupKFold splits to prevent leakage.
    
    Multiple rows can share the same interview_answer (different sub-questions
    from the same interview). Grouping by URL ensures all sub-questions from
    the same interview stay in the same fold.
    
    Args:
        dataset: HF Dataset with group_by column.
        n_splits: Number of folds.
        group_by: Column to group by (default: 'url').
        random_state: Random seed.
    
    Returns:
        List of fold dicts with train/val indices.
    """
    df = dataset.to_pandas()
    
    # Create groups
    groups = df[group_by].values
    
    # Use evasion_label for stratification guidance (though GroupKFold doesn't stratify)
    y = df["evasion_label_id"].values if "evasion_label_id" in df.columns else None
    
    # Create splits
    gkf = GroupKFold(n_splits=n_splits)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X=df, y=y, groups=groups)):
        split_info = {
            "fold": fold_idx,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
        }
        
        # Add label distribution info
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        split_info["train_clarity_dist"] = train_df["clarity_label"].value_counts().to_dict()
        split_info["val_clarity_dist"] = val_df["clarity_label"].value_counts().to_dict()
        split_info["train_evasion_dist"] = train_df["evasion_label"].value_counts().to_dict()
        split_info["val_evasion_dist"] = val_df["evasion_label"].value_counts().to_dict()
        
        splits.append(split_info)
    
    return splits


def create_stratified_splits(
    dataset: Dataset,
    n_splits: int = 5,
    stratify_by: str = "evasion_label_id",
    random_state: int = 42
) -> List[Dict]:
    """
    Create stratified splits (for comparison only - NOT recommended for final use).
    
    This ignores grouping and can leak interview_answer text across splits.
    Use only for quick iteration and comparison.
    
    Args:
        dataset: HF Dataset.
        n_splits: Number of folds.
        stratify_by: Column to stratify on.
        random_state: Random seed.
    
    Returns:
        List of fold dicts.
    """
    df = dataset.to_pandas()
    y = df[stratify_by].values
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X=df, y=y)):
        split_info = {
            "fold": fold_idx,
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "train_size": len(train_idx),
            "val_size": len(val_idx),
        }
        
        # Add label distribution info
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        split_info["train_clarity_dist"] = train_df["clarity_label"].value_counts().to_dict()
        split_info["val_clarity_dist"] = val_df["clarity_label"].value_counts().to_dict()
        split_info["train_evasion_dist"] = train_df["evasion_label"].value_counts().to_dict()
        split_info["val_evasion_dist"] = val_df["evasion_label"].value_counts().to_dict()
        
        splits.append(split_info)
    
    return splits


def save_splits(
    splits: List[Dict],
    output_path: str,
    split_name: str
):
    """
    Save split information to disk.
    
    Args:
        splits: List of split dicts.
        output_path: Directory to save to.
        split_name: Name for this split configuration.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{split_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "split_name": split_name,
            "n_splits": len(splits),
            "splits": splits
        }, f, indent=2)
    
    print(f"Saved splits to: {output_file}")


def load_splits(splits_path: str) -> List[Dict]:
    """
    Load saved splits.
    
    Args:
        splits_path: Path to splits JSON file.
    
    Returns:
        List of split dicts.
    """
    with open(splits_path, 'r') as f:
        data = json.load(f)
    return data["splits"]


def analyze_split_quality(splits: List[Dict], dataset: Dataset, group_by: str = "url"):
    """
    Analyze split quality and check for leakage.
    
    Args:
        splits: List of split dicts.
        dataset: Original dataset.
        group_by: Column used for grouping.
    """
    df = dataset.to_pandas()
    
    print("="*60)
    print("SPLIT QUALITY ANALYSIS")
    print("="*60)
    
    for fold_idx, split in enumerate(splits):
        print(f"\nFold {fold_idx}:")
        print(f"  Train: {split['train_size']:4d} examples")
        print(f"  Val:   {split['val_size']:4d} examples")
        
        # Check group overlap (should be 0 for grouped splits)
        train_groups = set(df.iloc[split['train_indices']][group_by].unique())
        val_groups = set(df.iloc[split['val_indices']][group_by].unique())
        overlap = train_groups & val_groups
        
        if overlap:
            print(f"  ⚠️  WARNING: {len(overlap)} overlapping groups!")
        else:
            print(f"  [OK] No group overlap")
        
        # Class balance
        print(f"  Clarity distribution (val): ", end="")
        for label, count in split['val_clarity_dist'].items():
            print(f"{label}: {count} ", end="")
        print()


def main():
    """Generate and save splits."""
    from load_dataset import load_clarity_dataset, normalize_labels, add_label_ids
    
    print("Loading dataset...")
    dataset = load_clarity_dataset()
    
    train = normalize_labels(dataset["train"])
    train = add_label_ids(train)
    
    print(f"\nTrain set: {len(train)} examples")
    print(f"Unique URLs: {len(set(train['url']))}")
    
    # Create grouped splits (recommended)
    print("\n" + "="*60)
    print("Creating GroupKFold splits (by URL)...")
    print("="*60)
    grouped_splits = create_grouped_splits(train, n_splits=5, group_by="url")
    
    # Save
    output_dir = os.environ.get("PROJECT_ROOT", ".") + "/artifacts/splits"
    save_splits(grouped_splits, output_dir, "grouped_5fold_by_url")
    
    # Analyze
    analyze_split_quality(grouped_splits, train, group_by="url")
    
    # Create stratified splits (for comparison)
    print("\n" + "="*60)
    print("Creating StratifiedKFold splits (for comparison only)...")
    print("="*60)
    stratified_splits = create_stratified_splits(train, n_splits=5)
    save_splits(stratified_splits, output_dir, "stratified_5fold_COMPARISON_ONLY")
    
    # Analyze
    analyze_split_quality(stratified_splits, train, group_by="url")
    
    print("\n" + "="*60)
    print("[OK] Split generation complete!")
    print("="*60)
    print(f"\nSaved to: {output_dir}/")
    print("  - grouped_5fold_by_url.json (USE THIS)")
    print("  - stratified_5fold_COMPARISON_ONLY.json (for quick iteration only)")


if __name__ == "__main__":
    main()
