#!/usr/bin/env python3
"""
Collect OOF predictions from DeBERTa-v3-large models for both tasks.
Compute per-model F1 scores to enable model selection.
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DebertaV2Tokenizer
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

TASK1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
EVASION_LABELS = [
    "Claims ignorance", 
    "Clarification", 
    "Declining to answer", 
    "Deflection", 
    "Dodging", 
    "Explicit", 
    "General", 
    "Implicit", 
    "Partial/half-answer"
]


class Task1Classifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}


class Task2Classifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}


def collect_task_oof(task_num, model_class, label_list):
    """Collect OOF predictions for a given task."""
    
    print(f"\n{'='*70}")
    print(f"Collecting OOF for Task {task_num}")
    print(f"{'='*70}")
    
    project_root = Path(__file__).parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    dataset = load_dataset("ailsntua/QEvasion")['train']
    n_samples = len(dataset)
    
    # Load fold splits
    with open(project_root / "artifacts/splits/grouped_5fold_by_url.json") as f:
        fold_data = json.load(f)
    
    # Initialize storage
    all_oof_logits = np.zeros((n_samples, len(label_list), 50))  # [samples, classes, models]
    all_oof_mask = np.zeros((n_samples, 50), dtype=bool)  # [samples, models]
    model_f1_scores = []
    
    # Find all model checkpoints
    all_model_dirs = sorted(project_root.glob(f"runs/task{task_num}_v3large_fold*"))
    
    # Filter to keep only the latest checkpoint per fold/seed
    from collections import defaultdict
    latest_dirs = defaultdict(lambda: (None, ""))
    
    for model_dir in all_model_dirs:
        # Parse fold and seed from directory name
        # Format: task1_v3large_fold0_seed42_20260115_213705
        parts = model_dir.name.split("_")
        fold = parts[2]  # e.g., "fold0"
        seed = parts[3]  # e.g., "seed42"
        timestamp = "_".join(parts[4:])  # e.g., "20260115_213705"
        
        key = (fold, seed)
        if timestamp > latest_dirs[key][1]:
            latest_dirs[key] = (model_dir, timestamp)
    
    model_dirs = sorted([d[0] for d in latest_dirs.values()])
    
    print(f"Found {len(all_model_dirs)} total directories, filtered to {len(model_dirs)} unique models")
    
    if len(model_dirs) != 50:
        print(f"WARNING: Expected 50 models, found {len(model_dirs)}")
    
    model_name = "microsoft/deberta-v3-large"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    
    for model_idx, model_dir in enumerate(tqdm(model_dirs, desc="Loading models")):
        # Parse fold and seed from directory name
        dir_name = model_dir.name
        fold = int(dir_name.split("fold")[1].split("_")[0])
        
        # Load checkpoint
        ckpt_path = model_dir / "best_model.pt"
        if not ckpt_path.exists():
            print(f"WARNING: Checkpoint not found: {ckpt_path}")
            continue
        
        # Load model
        model = model_class(model_name, num_labels=len(label_list))
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Get validation indices for this fold
        val_indices = fold_data['splits'][fold]['val_indices']
        
        # Prepare validation data
        label_field = 'clarity_label' if task_num == 1 else 'evasion_label'
        val_texts = [f"Question: {dataset[i]['question']}\nAnswer: {dataset[i]['interview_answer']}" 
                     for i in val_indices]
        val_labels = [label_list.index(dataset[i][label_field]) for i in val_indices]
        
        # Predict
        batch_size = 8  # Reduced from 16 to save memory
        val_logits = []
        
        with torch.no_grad():
            for i in range(0, len(val_texts), batch_size):
                batch_texts = val_texts[i:i+batch_size]
                encoding = tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_logits.append(outputs['logits'].cpu().numpy())
        
        val_logits = np.vstack(val_logits)
        val_preds = val_logits.argmax(axis=1)
        
        # Compute F1
        model_f1 = f1_score(val_labels, val_preds, 
                           labels=list(range(len(label_list))), 
                           average='macro')
        model_f1_scores.append({
            'model_idx': model_idx,
            'fold': fold,
            'dir': str(model_dir),
            'f1': model_f1
        })
        
        # Store OOF predictions
        for local_idx, global_idx in enumerate(val_indices):
            all_oof_logits[global_idx, :, model_idx] = val_logits[local_idx]
            all_oof_mask[global_idx, model_idx] = True
        
        # Save intermediate progress every 10 models
        if (model_idx + 1) % 10 == 0:
            output_dir = project_root / "artifacts/oof"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.savez(
                output_dir / f"task{task_num}_v3large_oof_partial.npz",
                logits=all_oof_logits,
                mask=all_oof_mask,
                model_count=model_idx + 1
            )
            print(f"  Saved intermediate results ({model_idx + 1}/{len(model_dirs)} models)")
        
        del model, checkpoint, val_logits
        torch.cuda.empty_cache()
    
    # Compute ensemble OOF predictions
    print("\nComputing ensemble OOF predictions...")
    print(f"all_oof_logits shape: {all_oof_logits.shape}  # Should be (samples, classes, models)")
    ensemble_logits = np.zeros((n_samples, len(label_list)))
    for i in range(n_samples):
        mask = all_oof_mask[i]
        if mask.sum() > 0:
            # CRITICAL FIX: all_oof_logits[i, :, mask] gives (n_models_selected, n_classes)
            # So we need to average across axis=0 (models) to get (n_classes,)
            ensemble_logits[i] = all_oof_logits[i, :, mask].mean(axis=0)
    
    ensemble_preds = ensemble_logits.argmax(axis=1)
    
    # Compute ground truth
    label_field = 'clarity_label' if task_num == 1 else 'evasion_label'
    ground_truth = [label_list.index(dataset[i][label_field]) for i in range(n_samples)]
    
    # Compute ensemble F1
    ensemble_f1 = f1_score(ground_truth, ensemble_preds,
                          labels=list(range(len(label_list))),
                          average='macro')
    
    print(f"\n{'='*70}")
    print(f"Task {task_num} OOF Results:")
    print(f"  Ensemble F1: {ensemble_f1:.4f}")
    print(f"  Per-model F1 range: {min(m['f1'] for m in model_f1_scores):.4f} - {max(m['f1'] for m in model_f1_scores):.4f}")
    print(f"  Mean model F1: {np.mean([m['f1'] for m in model_f1_scores]):.4f}")
    print(f"{'='*70}\n")
    
    # Save results
    output_dir = project_root / "artifacts/oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_dir / f"task{task_num}_v3large_oof.npz",
        logits=all_oof_logits,
        mask=all_oof_mask,
        ensemble_logits=ensemble_logits,
        ensemble_preds=ensemble_preds,
        ground_truth=ground_truth,
        ensemble_f1=ensemble_f1
    )
    
    # Save model scores
    with open(output_dir / f"task{task_num}_v3large_model_scores.json", 'w') as f:
        json.dump({
            'models': model_f1_scores,
            'ensemble_f1': float(ensemble_f1),
            'mean_f1': float(np.mean([m['f1'] for m in model_f1_scores]))
        }, f, indent=2)
    
    print(f"Saved to {output_dir}/task{task_num}_v3large_oof.npz")
    
    return ensemble_f1, model_f1_scores


def main():
    # Task 1
    task1_f1, task1_scores = collect_task_oof(1, Task1Classifier, TASK1_LABELS)
    
    # Task 2
    task2_f1, task2_scores = collect_task_oof(2, Task2Classifier, EVASION_LABELS)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Task 1 v3-large OOF F1: {task1_f1:.4f}")
    print(f"Task 2 v3-large OOF F1: {task2_f1:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
