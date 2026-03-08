#!/usr/bin/env python3
"""
10-Seed Training for Task 1 or Task 2

Supports both tasks with --task argument:
- Task 1: 3-way classification (CLARITY)
- Task 2: 9-way classification (EVASION)
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
import random

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

TASK1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"
]


def set_seed(seed):
    """Set all random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TaskClassifier(nn.Module):
    def __init__(self, model_name, num_labels, label_smoothing=0.0):
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
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_single_seed(task, fold, seed, model_name, batch_size=4, grad_accum=8, epochs=6, lr=1e-5, label_smoothing=0.03):
    """Train one fold with one seed for specified task."""
    
    set_seed(seed)
    
    labels = TASK1_LABELS if task == 1 else TASK2_LABELS
    num_labels = len(labels)
    label_column = 'clarity_label' if task == 1 else 'evasion_label'
    
    print(f"\n{'='*60}")
    print(f"TASK {task} 10-SEED - FOLD {fold}, SEED {seed}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    dataset = load_dataset("ailsntua/QEvasion")
    train_data = dataset['train']
    
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "artifacts/splits/grouped_5fold_by_url.json") as f:
        fold_data = json.load(f)
    
    train_indices = fold_data['splits'][fold]['train_indices']
    val_indices = fold_data['splits'][fold]['val_indices']
    
    all_texts = [f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['interview_answer']}" 
                 for i in range(len(train_data))]
    all_labels = [labels.index(train_data[i][label_column]) for i in range(len(train_data))]
    
    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_texts = [all_texts[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TaskClassifier(model_name=model_name, num_labels=num_labels, label_smoothing=label_smoothing)
    model = model.to(device)
    
    train_dataset = SimpleDataset(train_texts, train_labels, tokenizer)
    val_dataset = SimpleDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    
    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training
    best_f1 = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / f"runs/task{task}_10seed_fold{fold}_seed{seed}_{timestamp}"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
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
            
            total_loss += loss.item() * grad_accum
            pbar.set_postfix({'loss': f'{total_loss/(step+1):.4f}'})
        
        # Validation
        model.eval()
        all_preds = []
        all_true = []
        all_logits = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                preds = logits.argmax(dim=-1).cpu()
                
                all_preds.extend(preds.tolist())
                all_true.extend(labels.tolist())
                all_logits.append(logits.cpu().numpy())
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_true, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Val Macro-F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold,
                'seed': seed,
                'epoch': epoch,
                'val_f1': val_f1,
            }, save_dir / 'best_model.pt')
            
            np.save(save_dir / 'oof_logits.npy', np.vstack(all_logits))
            np.save(save_dir / 'oof_labels.npy', np.array(all_true))
            
            print(f"[OK] New best model saved!")
    
    print(f"\nBest Val Macro-F1: {best_f1:.4f}")
    print(f"Saved to: {save_dir}")
    
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, required=True, choices=[1, 2], help="Task number (1 or 2)")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-xlarge")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    args = parser.parse_args()
    
    train_single_seed(
        task=args.task,
        fold=args.fold,
        seed=args.seed,
        model_name=args.model_name,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation_steps,
        epochs=args.epochs,
        lr=args.lr,
        label_smoothing=args.label_smoothing
    )


if __name__ == "__main__":
    main()
