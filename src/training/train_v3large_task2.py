#!/usr/bin/env python3
"""
Train DeBERTa-v3-large 10-seed ensemble for Task 2

9-way classification for evasion types
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import f1_score

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


class Task2Classifier(nn.Module):
    def __init__(self, model_name, num_labels=9, label_smoothing=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.label_smoothing = label_smoothing
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        
        return {"logits": logits, "loss": loss}


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


def train_single_seed(fold, seed):
    """Train single fold with given seed."""
    
    print(f"\n{'='*70}")
    print(f"TRAINING: Fold {fold}, Seed {seed}")
    print(f"{'='*70}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/deberta-v3-large"
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load data
    dataset = load_dataset("ailsntua/QEvasion")
    train_data = dataset['train']
    
    project_root = Path(__file__).parent.parent.parent  # src/training/script.py -> project root
    with open(project_root / "artifacts/splits/grouped_5fold_by_url.json") as f:
        fold_data = json.load(f)
    
    train_indices = fold_data['splits'][fold]['train_indices']
    val_indices = fold_data['splits'][fold]['val_indices']
    
    all_texts = [f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['interview_answer']}" 
                 for i in range(len(train_data))]
    all_labels = [EVASION_LABELS.index(train_data[i]['evasion_label']) for i in range(len(train_data))]
    
    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_texts = [all_texts[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Task2Classifier(model_name=model_name, num_labels=9, label_smoothing=0.03)
    model = model.to(device)
    
    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
    val_dataset = SimpleDataset(val_texts, val_labels, tokenizer)
    
    batch_size = 4
    grad_accum = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    num_epochs = 6
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )
    
    # Training
    best_f1 = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss'] / grad_accum
            
            loss.backward()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * grad_accum
        
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                val_preds.extend(outputs['logits'].argmax(dim=1).cpu().numpy())
        
        val_f1 = f1_score(val_labels, val_preds, labels=list(range(9)), average='macro')
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch + 1
            
            # Save best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = project_root / f"runs/task2_v3large_fold{fold}_seed{seed}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_f1': best_f1,
                'fold': fold,
                'seed': seed,
            }, save_dir / 'best_model.pt')
            
            print(f"-> Saved best model (F1={best_f1:.4f})")
    
    print(f"\n{'='*70}")
    print(f"Best Val F1: {best_f1:.4f} (epoch {best_epoch})")
    print(f"{'='*70}\n")
    
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    
    best_f1 = train_single_seed(args.fold, args.seed)
    print(f"\nFinal: Fold {args.fold}, Seed {args.seed} -> F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
