"""
Training utilities for CLARITY task.

Implements training loop with:
- Per-fold class weighting
- Truncation logging
- Grouped CV support
- Run registry integration
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add parent dirs to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.load_dataset import EVASION_LABELS
from metrics.compute_metrics import compute_all_metrics
from training.run_registry import RunRegistry


class ClarityDataset(Dataset):
    """PyTorch dataset for CLARITY task."""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int = 512,
        text_pairs: Optional[List[str]] = None
    ):
        """
        Args:
            texts: List of input texts (or sequence A for pair encoding)
            labels: Label indices
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            text_pairs: Optional list of text pairs (sequence B for pair encoding)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_pairs = text_pairs
        self.use_pair_encoding = text_pairs is not None
        
        # Track truncation
        self.truncation_count = 0
        self.token_lengths = []
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize with or without pairs
        if self.use_pair_encoding:
            text_pair = self.text_pairs[idx]
            
            # Pre-tokenize question context to ensure it's not too long
            # Reserve at least 256 tokens for the answer
            max_question_length = min(256, self.max_length // 2)
            question_encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=max_question_length,
                add_special_tokens=False  # We'll add them in the full encoding
            )
            
            # Decode back to text (truncated if needed)
            truncated_question = self.tokenizer.decode(
                question_encoding['input_ids'],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Now encode the pair with the truncated question
            encoding = self.tokenizer(
                truncated_question,
                text_pair,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,  # Truncate both if needed (mostly answer)
                return_tensors="pt"
            )
        else:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        # Track truncation
        input_ids = encoding["input_ids"].squeeze()
        actual_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        self.token_lengths.append(actual_length)
        
        if actual_length >= self.max_length:
            self.truncation_count += 1
        
        result = {
            "input_ids": input_ids,
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }
        
        # Add token_type_ids if present (some models like DeBERTa use it)
        if "token_type_ids" in encoding:
            result["token_type_ids"] = encoding["token_type_ids"].squeeze()
        
        return result
    
    def get_truncation_stats(self) -> Dict:
        """Get truncation statistics."""
        if not self.token_lengths:
            return {}
        
        return {
            "truncation_count": self.truncation_count,
            "truncation_rate": self.truncation_count / len(self),
            "mean_tokens": np.mean(self.token_lengths),
            "max_tokens": np.max(self.token_lengths),
            "min_tokens": np.min(self.token_lengths),
            "total_examples": len(self),
            "encoding_mode": "pair" if self.use_pair_encoding else "concat"
        }


def compute_class_weights(labels: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute class weights inversely proportional to class frequencies.
    
    Args:
        labels: Label array
        device: Device for tensor
    
    Returns:
        Class weights tensor
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    
    # Ensure we have weights for all classes (even if not in this fold)
    full_weights = np.ones(len(EVASION_LABELS))
    full_weights[classes] = weights
    
    return torch.tensor(full_weights, dtype=torch.float32, device=device)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    class_weights: Optional[torch.Tensor] = None,
    gradient_accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device
        class_weights: Class weights (optional)
        gradient_accumulation_steps: Accumulate gradients over N steps
    
    Returns:
        Dict with training metrics
    """
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(pbar):
        # Move to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            class_weights=class_weights
        )
        
        loss = outputs["loss"]
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward
        loss.backward()
        
        # Update weights every N steps
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Track metrics (use unscaled loss for reporting)
        total_loss += loss.item() * gradient_accumulation_steps
        preds = torch.argmax(outputs["logits"], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_all_metrics(all_labels, all_preds, EVASION_LABELS, prefix="train_")
    metrics["train_loss"] = avg_loss
    
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate model.
    
    Args:
        model: Model to evaluate
        dataloader: Eval dataloader
        device: Device
    
    Returns:
        Tuple of (metrics, predictions, true_labels)
    """
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Evaluating")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        total_loss += outputs["loss"].item()
        preds = torch.argmax(outputs["logits"], dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    metrics = compute_all_metrics(all_labels, all_preds, EVASION_LABELS, prefix="val_")
    metrics["val_loss"] = avg_loss
    
    return metrics, all_preds, all_labels


def save_checkpoint(
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    checkpoint_path: Path
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer
        scheduler: Scheduler
        epoch: Current epoch
        metrics: Current metrics
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": metrics
    }
    
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer (optional)
        scheduler: Scheduler (optional)
    
    Returns:
        Checkpoint dict
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint


if __name__ == "__main__":
    print("Training utilities loaded successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")


def compute_class_weights_sqrt(labels: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Compute class weights using sqrt of inverse frequency.
    
    Less aggressive than balanced weights, better for noisy labels.
    
    Args:
        labels: Label array
        device: Device for tensor
    
    Returns:
        Class weights tensor
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(EVASION_LABELS)
    
    weights = np.ones(n_classes)
    for label_id, count in counts.items():
        # sqrt of inverse frequency
        weights[label_id] = np.sqrt(total / (n_classes * count))
    
    # Normalize to have mean = 1
    weights = weights / weights.mean()
    
    return torch.tensor(weights, dtype=torch.float32, device=device)


def get_logit_adjustment_bias(labels: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Compute logit adjustment bias for long-tailed classification.
    
    bias_k = tau * log(p_k) where p_k is class prior
    
    Args:
        labels: Training labels
        tau: Temperature scaling (higher = more adjustment)
    
    Returns:
        Bias vector to add to logits
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(EVASION_LABELS)
    
    priors = np.zeros(n_classes)
    for label_id, count in counts.items():
        priors[label_id] = count / total
    
    # Add small epsilon to avoid log(0)
    priors = np.clip(priors, 1e-7, 1.0)
    
    bias = tau * np.log(priors)
    return bias
