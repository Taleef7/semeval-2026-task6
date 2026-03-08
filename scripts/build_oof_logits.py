#!/usr/bin/env python3
"""
Phase 2.1: Build Out-of-Fold (OOF) logits for ensemble calibration.

For each fold model:
- Run inference on its validation split
- Save logits for calibration and stacking
"""

import json
import numpy as np
import torch
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer
from models.encoder_classifier import EvasionClassifier
from data.load_dataset import load_clarity_dataset, normalize_labels, add_label_ids, EVASION_LABELS
from data.preprocess import prepare_model_inputs


def load_checkpoint(checkpoint_path, model_name, device):
    """Load a fold checkpoint."""
    model = EvasionClassifier(model_name=model_name, num_labels=9)
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def get_logits(model, tokenizer, texts, device, batch_size=32, max_length=512):
    """Get logits for a list of texts."""
    all_logits = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting logits"):
        batch_texts = texts[i:i+batch_size]
        
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"].cpu().numpy()
        
        all_logits.append(logits)
    
    return np.concatenate(all_logits, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    model_name = "microsoft/deberta-v3-base"
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_clarity_dataset()
    train_full = normalize_labels(dataset["train"])
    train_full = add_label_ids(train_full)
    train_full = prepare_model_inputs(train_full)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load splits
    print("Loading splits...")
    with open("artifacts/splits/grouped_5fold_by_url.json") as f:
        splits_data = json.load(f)
    
    # Find 6-epoch checkpoints
    checkpoint_base = Path("runs")
    fold_dirs = sorted([d for d in checkpoint_base.iterdir() 
                       if d.is_dir() and "20251228_135055" in d.name])
    
    if len(fold_dirs) < 5:
        # Try to find by pattern
        print("Looking for 6-epoch checkpoints...")
        fold_dirs = []
        for fold_idx in range(5):
            pattern = f"*fold{fold_idx}*seed42*"
            matches = list(checkpoint_base.glob(pattern))
            # Filter for the right ones (not 8-epoch)
            for m in matches:
                if "20251228" in m.name:
                    fold_dirs.append(m)
                    break
    
    print(f"Found {len(fold_dirs)} checkpoint directories")
    
    # Initialize OOF arrays
    n_samples = len(train_full)
    oof_logits = np.zeros((n_samples, 9), dtype=np.float32)
    oof_labels = np.array(train_full["evasion_label_id"])
    oof_indices = np.zeros(n_samples, dtype=np.int32)
    
    for fold_idx, split_info in enumerate(splits_data["splits"]):
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx}")
        print(f"{'='*60}")
        
        val_indices = split_info["val_indices"]
        
        # Find checkpoint for this fold
        checkpoint_path = None
        for d in checkpoint_base.iterdir():
            if f"fold{fold_idx}" in d.name and "20251228" in d.name and "seed42" in d.name:
                cp = d / "best_model.pt"
                if cp.exists():
                    checkpoint_path = cp
                    break
        
        if checkpoint_path is None:
            print(f"WARNING: No checkpoint found for fold {fold_idx}")
            continue
        
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load model
        model = load_checkpoint(checkpoint_path, model_name, device)
        
        # Get validation texts
        val_data = train_full.select(val_indices)
        val_texts = val_data["model_input"]
        
        # Get logits
        logits = get_logits(model, tokenizer, val_texts, device)
        
        # Store in OOF array
        for local_idx, global_idx in enumerate(val_indices):
            oof_logits[global_idx] = logits[local_idx]
            oof_indices[global_idx] = fold_idx
        
        print(f"  Processed {len(val_indices)} validation samples")
        
        # Clear GPU memory
        del model
        torch.cuda.empty_cache()
    
    # Save OOF logits
    output_dir = Path("artifacts/oof")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "oof_logits_6epoch.npy", oof_logits)
    np.save(output_dir / "oof_labels.npy", oof_labels)
    np.save(output_dir / "oof_fold_indices.npy", oof_indices)
    
    # Compute OOF metrics
    from sklearn.metrics import f1_score
    oof_preds = np.argmax(oof_logits, axis=1)
    oof_f1 = f1_score(oof_labels, oof_preds, average="macro")
    
    print(f"\n{'='*60}")
    print(f"OOF RESULTS")
    print(f"{'='*60}")
    print(f"OOF Macro-F1: {oof_f1:.4f}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
