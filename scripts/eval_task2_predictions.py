#!/usr/bin/env python3
"""
Generate Task 2 predictions on evaluation dataset

Two modes:
1. Gold (conservative): Use distillation-based 0.48 model
2. Hierarchical: Use hierarchical ensemble with Task 1 masking
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader

EVASION_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"
]

T1_TO_T2_MASK = {
    "Clear Reply": [0],  # Only Explicit
    "Ambivalent": [1, 2, 3, 4, 5],  # Implicit, Dodging, General, Deflection, Partial
    "Clear Non-Reply": [6, 7, 8],  # Declining, Ignorance, Clarification
}


class Task2Classifier(nn.Module):
    def __init__(self, model_name, num_labels=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return {"logits": logits}


class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


def load_task1_predictions(task1_pred_file):
    """Load Task 1 predictions from prediction file."""
    with open(task1_pred_file) as f:
        return [line.strip() for line in f]


def apply_hierarchical_masking(logits, task1_pred):
    """Apply hierarchical constraint based on Task 1 prediction."""
    mask = torch.full((len(EVASION_LABELS),), float('-inf'))
    allowed_indices = T1_TO_T2_MASK[task1_pred]
    mask[allowed_indices] = 0.0
    return logits + mask


def generate_gold_predictions(eval_df, output_dir):
    """Generate predictions using gold distillation model (0.48)."""
    print("\n" + "="*60)
    print("MODE: GOLD (Conservative)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/deberta-xlarge"
    
    project_root = Path(__file__).parent.parent
    
    # Prepare texts
    eval_texts = [f"Question: {row['question']}\nAnswer: {row['interview_answer']}" 
                  for _, row in eval_df.iterrows()]
    
    # Find gold model checkpoints (5 folds, distillation-based)
    gold_models = []
    for fold in range(5):
        model_dirs = sorted((project_root / "runs").glob(f"distillation_fold{fold}_20260108*"))
        if len(model_dirs) > 0:
            gold_models.append(model_dirs[-1] / 'best_model.pt')
            print(f"Fold {fold}: {model_dirs[-1].name}")
    
    if len(gold_models) != 5:
        raise ValueError(f"Found {len(gold_models)} Task 2 gold checkpoints, expected 5")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset = SimpleDataset(eval_texts, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    ensemble_logits = np.zeros((len(eval_texts), len(EVASION_LABELS)))
    
    for fold, ckpt_path in enumerate(gold_models):
        print(f"\nProcessing Fold {fold}...")
        
        model = Task2Classifier(model_name=model_name, num_labels=9)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        batch_logits = []
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc=f"Fold {fold}", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_logits.append(outputs['logits'].cpu().numpy())
        
        ensemble_logits += np.vstack(batch_logits)
        del model
        torch.cuda.empty_cache()
    
    ensemble_logits /= len(gold_models)
    predictions = [EVASION_LABELS[idx] for idx in ensemble_logits.argmax(axis=1)]
    
    return predictions


def generate_hierarchical_predictions(eval_df, task1_pred_file, output_dir):
    """Generate predictions using hierarchical ensemble."""
    print("\n" + "="*60)
    print("MODE: HIERARCHICAL (Novel)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/deberta-xlarge"
    
    project_root = Path(__file__).parent.parent
    
    # Load Task 1 predictions
    task1_preds = load_task1_predictions(task1_pred_file)
    print(f"Loaded {len(task1_preds)} Task 1 predictions")
    
    # Prepare texts
    eval_texts = [f"Question: {row['question']}\nAnswer: {row['interview_answer']}" 
                  for _, row in eval_df.iterrows()]
    
    # Find pure CE model checkpoints (15 models: 5 folds × 3 seeds)
    models_by_fold = {}
    for fold in range(5):
        models_by_fold[fold] = {}
        for seed_offset in range(3):
            seed = 42 + seed_offset
            dirs = sorted((project_root / "runs").glob(f"pure_ce_fold{fold}_seed{seed}_*"))
            if len(dirs) > 0:
                models_by_fold[fold][seed] = dirs[-1] / 'best_model.pt'
                print(f"Fold {fold} Seed {seed}: {dirs[-1].name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset = SimpleDataset(eval_texts, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    ensemble_logits = np.zeros((len(eval_texts), len(EVASION_LABELS)))
    
    # Process each fold
    for fold in range(5):
        print(f"\nProcessing Fold {fold}...")
        fold_logits = np.zeros((len(eval_texts), len(EVASION_LABELS)))
        
        # Average across seeds within fold
        for seed, ckpt_path in sorted(models_by_fold[fold].items()):
            model = Task2Classifier(model_name=model_name, num_labels=9)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            batch_logits = []
            with torch.no_grad():
                for batch in tqdm(eval_loader, desc=f"Fold {fold} Seed {seed}", leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_logits.append(outputs['logits'].cpu().numpy())
            
            fold_logits += np.vstack(batch_logits)
            del model
            torch.cuda.empty_cache()
        
        fold_logits /= len(models_by_fold[fold])
        ensemble_logits += fold_logits
    
    ensemble_logits /= 5
    
    # Apply hierarchical masking
    print("\nApplying hierarchical masking...")
    final_predictions = []
    masked_count = 0
    
    for i in range(len(eval_texts)):
        logits = torch.tensor(ensemble_logits[i])
        task1_pred = task1_preds[i]
        
        masked_logits = apply_hierarchical_masking(logits, task1_pred)
        pred_idx = masked_logits.argmax().item()
        final_predictions.append(EVASION_LABELS[pred_idx])
        
        if pred_idx != logits.argmax().item():
            masked_count += 1
    
    print(f"Hierarchical masking changed {masked_count}/{len(eval_texts)} predictions ({100*masked_count/len(eval_texts):.1f}%)")
    
    return final_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["gold", "hierarchical"])
    parser.add_argument("--task1_predictions", type=str, help="Path to Task 1 prediction file (required for hierarchical)")
    args = parser.parse_args()
    
    print("="*60)
    print("TASK 2 EVALUATION PREDICTIONS")
    print("="*60)
    
    project_root = Path(__file__).parent.parent
    eval_df = pd.read_csv(project_root / "clarity_task_evaluation_dataset.csv")
    print(f"\nEvaluation samples: {len(eval_df)}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / f"submissions/eval_task2_{args.mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "gold":
        predictions = generate_gold_predictions(eval_df, output_dir)
    else:  # hierarchical
        if not args.task1_predictions:
            raise ValueError("--task1_predictions required for hierarchical mode")
        predictions = generate_hierarchical_predictions(eval_df, args.task1_predictions, output_dir)
    
    # Save
    with open(output_dir / "prediction", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    import shutil
    shutil.make_archive(str(output_dir / "task2_submission"), 'zip', output_dir, "prediction")
    
    from collections import Counter
    dist = Counter(predictions)
    
    print(f"\n{'='*60}")
    print("TASK 2 EVAL SUBMISSION CREATED")
    print(f"{'='*60}")
    print(f"Directory: {output_dir}")
    print(f"Zip: {output_dir}/task2_submission.zip")
    print(f"\nPrediction distribution:")
    for label in EVASION_LABELS:
        print(f"  {label:25s}: {dist.get(label, 0):4d} ({100*dist.get(label, 0)/len(predictions):.1f}%)")


if __name__ == "__main__":
    main()
