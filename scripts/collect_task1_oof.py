#!/usr/bin/env python3
"""
Generate OOF predictions from trained Task 1 models

Since OOF wasn't saved during training, we regenerate it by loading
each model and predicting on its validation fold.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score, classification_report

TASK1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


class Task1Classifier(nn.Module):
    def __init__(self, model_name, num_labels=3):
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
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
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
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': self.labels[idx]
        }


def main():
    print("="*60)
    print("GENERATING TASK 1 OOF PREDICTIONS")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/deberta-xlarge"
    
    project_root = Path(__file__).parent.parent
    
    # Load data
    dataset = load_dataset("ailsntua/QEvasion")
    train_data = dataset['train']
    n_samples = len(train_data)
    
    # Load fold splits
    with open(project_root / "artifacts/splits/grouped_5fold_by_url.json") as f:
        fold_data = json.load(f)
    
    # Prepare all texts and labels
    all_texts = [f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['interview_answer']}" 
                 for i in range(n_samples)]
    all_labels = np.array([TASK1_LABELS.index(train_data[i]['clarity_label']) for i in range(n_samples)])
    
    # Find all 50 models
    model_dirs = sorted((project_root / "runs").glob("task1_10seed_fold*_seed*"))
    model_paths = [(d, d / 'best_model.pt') for d in model_dirs if (d / 'best_model.pt').exists()]
    
    print(f"\nFound {len(model_paths)} trained models")
    
    # Initialize arrays
    all_oof_logits = np.zeros((n_samples, len(TASK1_LABELS), len(model_paths)))
    all_oof_mask = np.zeros((n_samples, len(model_paths)), dtype=bool)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Process each model
    for model_idx, (model_dir, model_path) in enumerate(tqdm(model_paths, desc="Models")):
        # Parse fold from directory name
        fold = int(model_dir.name.split('_fold')[1].split('_')[0])
        
        # Get validation indices for this fold
        val_indices = fold_data['splits'][fold]['val_indices']
        
        # Prepare validation data
        val_texts = [all_texts[i] for i in val_indices]
        val_labels = [all_labels[i] for i in val_indices]
        
        # Load model
        model = Task1Classifier(model_name=model_name, num_labels=3)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Create dataset and loader
        val_dataset = SimpleDataset(val_texts, val_labels, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Predict
        val_logits = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_logits.append(outputs['logits'].cpu().numpy())
        
        val_logits = np.vstack(val_logits)
        
        # Store in full array
        all_oof_logits[val_indices, :, model_idx] = val_logits
        all_oof_mask[val_indices, model_idx] = True
        
        del model
        torch.cuda.empty_cache()
    
    # Compute ensemble predictions
    # For each sample, average across models where it was validation
    ensemble_logits = np.zeros((n_samples, len(TASK1_LABELS)))
    for i in range(n_samples):
        mask = all_oof_mask[i]
        if mask.sum() > 0:
            # When indexing with mask, shape is (n_models_true, n_classes)
            # So average across axis=0 (models dimension)
            ensemble_logits[i] = all_oof_logits[i, :, mask].mean(axis=0)
    
    ensemble_preds = ensemble_logits.argmax(axis=1)
    
    # Compute OOF F1 (only on samples that were validation in at least one fold)
    val_mask = all_oof_mask.any(axis=1)
    
    oof_f1 = f1_score(
        all_labels[val_mask],
        ensemble_preds[val_mask],
        labels=[0, 1, 2],
        average='macro'
    )
    
    print(f"\n{'='*60}")
    print(f"Ensemble OOF F1: {oof_f1:.4f}")
    print(f"{'='*60}")
    
    # Save
    output_dir = project_root / "artifacts/oof"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "task1_10seed_oof.npz"
    np.savez_compressed(
        output_file,
        logits=all_oof_logits,
        labels=all_labels,
        val_mask=val_mask,
        ensemble_logits=ensemble_logits,
        ensemble_preds=ensemble_preds,
        label_names=TASK1_LABELS
    )
    
    print(f"\nSaved to: {output_file}")
    
    # Print classification report
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(
        all_labels[val_mask],
        ensemble_preds[val_mask],
        labels=[0, 1, 2],
        target_names=TASK1_LABELS,
        digits=4
    ))


if __name__ == "__main__":
    main()
