"""
Stratified Group K-Fold splitter for CLARITY task.

Extends GroupKFold to approximately stratify by labels while respecting group boundaries.
This prevents wildly imbalanced label distributions per fold while maintaining zero leakage.
"""

import numpy as np
from typing import Iterator, Tuple
from collections import Counter, defaultdict


class StratifiedGroupKFold:
    """
    Stratified Group K-Fold cross-validator.
    
    Provides train/test indices to split data while:
    1. Keeping all rows with the same group together (prevents leakage)
    2. Approximately balancing label distributions across folds
    
    This is critical for macro-F1 evaluation on imbalanced datasets.
    """
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle groups before splitting.
            random_state: Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(
        self,
        X,
        y,
        groups
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X: Training data (can be None, only used for length).
            y: Target variable for stratification.
            groups: Group labels for samples.
        
        Yields:
            Tuple of (train_indices, test_indices) for each fold.
        """
        n_samples = len(y) if hasattr(y, '__len__') else len(X)
        
        # Convert to numpy arrays
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        # Get unique groups and their labels
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if n_groups < self.n_splits:
            raise ValueError(
                f"Number of groups ({n_groups}) is less than n_splits ({self.n_splits})"
            )
        
        # For each group, get the most common label (for stratification purposes)
        group_to_label = {}
        for group in unique_groups:
            group_mask = groups == group
            group_labels = y[group_mask]
            # Most common label in this group
            most_common = Counter(group_labels).most_common(1)[0][0]
            group_to_label[group] = most_common
        
        # Group groups by their most common label
        label_to_groups = defaultdict(list)
        for group, label in group_to_label.items():
            label_to_groups[label].append(group)
        
        # Shuffle groups within each label
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            for label in label_to_groups:
                rng.shuffle(label_to_groups[label])
        
        # Distribute groups to folds in a round-robin fashion within each label
        fold_groups = [[] for _ in range(self.n_splits)]
        
        for label, label_groups in label_to_groups.items():
            for fold_idx, group in enumerate(label_groups):
                fold_groups[fold_idx % self.n_splits].append(group)
        
        # Generate train/test indices for each fold
        for test_fold_idx in range(self.n_splits):
            test_groups = set(fold_groups[test_fold_idx])
            train_groups = set()
            for i in range(self.n_splits):
                if i != test_fold_idx:
                    train_groups.update(fold_groups[i])
            
            # Convert group sets to sample indices
            train_indices = np.where(np.isin(groups, list(train_groups)))[0]
            test_indices = np.where(np.isin(groups, list(test_groups)))[0]
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the number of splitting iterations."""
        return self.n_splits


def test_stratified_group_kfold():
    """Test the splitter with a small example."""
    import pandas as pd
    
    # Create test data
    data = pd.DataFrame({
        'group': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'E', 'E', 'F', 'F', 'G', 'G'],
        'label': [0,    0,    0,    1,    1,    0,    0,    0,    1,    1,    2,    2,    1,    1,    2,    2]
    })
    
    sgkf = StratifiedGroupKFold(n_splits=3, random_state=42)
    
    print("Testing StratifiedGroupKFold:")
    print(f"Total samples: {len(data)}")
    print(f"Unique groups: {data['group'].nunique()}")
    print(f"Label distribution: {dict(Counter(data['label']))}")
    print()
    
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X=None, y=data['label'], groups=data['group'])):
        train_df = data.iloc[train_idx]
        val_df = data.iloc[val_idx]
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_idx)} samples, groups: {sorted(train_df['group'].unique())}")
        print(f"  Val:   {len(val_idx)} samples, groups: {sorted(val_df['group'].unique())}")
        print(f"  Train labels: {dict(Counter(train_df['label']))}")
        print(f"  Val labels:   {dict(Counter(val_df['label']))}")
        
        # Check no group overlap
        overlap = set(train_df['group']) & set(val_df['group'])
        print(f"  Group overlap: {len(overlap)} (should be 0)")
        print()


if __name__ == "__main__":
    test_stratified_group_kfold()
