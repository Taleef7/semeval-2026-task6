#!/usr/bin/env python3
"""
Task 1: DeBERTa-xlarge Direct Training with Distillation

Per advisor commands v6:
- Train DeBERTa-xlarge directly on Task 1 (3-way classification)
- Map 9-way teacher distribution to 3-way parent distribution for distillation
- Target: 0.63-0.65 (currently 0.62)

3-way Hierarchy:
- Direct: Explicit, Implicit
- Indirect: Dodging, General, Deflection, Partial/half-answer
- Non-answer: Declining to answer, Claims ignorance, Clarification
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

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# 3-way labels for Task 1
TASK1_LABELS = ["Direct", "Indirect", "Non-answer"]
TASK1_LABEL2ID = {l: i for i, l in enumerate(TASK1_LABELS)}

# 9-way to 3-way mapping
TASK2_TO_TASK1 = {
    "Explicit": "Direct",
    "Implicit": "Direct",
    "Dodging": "Indirect",
    "General": "Indirect",
    "Deflection": "Indirect",
    "Partial/half-answer": "Indirect",
    "Declining to answer": "Non-answer",
    "Claims ignorance": "Non-answer",
    "Clarification": "Non-answer"
}

# Indices for mapping 9-way probs to 3-way
NINE_TO_THREE_MAPPING = {
    0: 0,  # Explicit -> Direct
    1: 0,  # Implicit -> Direct
    2: 1,  # Dodging -> Indirect
    3: 1,  # General -> Indirect
    4: 1,  # Deflection -> Indirect
    5: 1,  # Partial -> Indirect
    6: 2,  # Declining -> Non-answer
    7: 2,  # Claims ignorance -> Non-answer
    8: 2,  # Clarification -> Non-answer
}


class Task1Classifier(nn.Module):
    """Simple 3-class classifier for Task 1."""
    
    def __init__(self, model_name, num_labels=3, label_smoothing=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.label_smoothing = label_smoothing
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(pooled)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        
        return {"logits": logits, "loss": loss}


class Task1Dataset(Dataset):
    """Dataset for Task 1 with optional soft labels."""
    
    def __init__(self, texts, hard_labels, soft_labels, tokenizer, max_length=512):
        self.texts = texts
        self.hard_labels = hard_labels
        self.soft_labels = soft_labels
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
            'hard_label': torch.tensor(self.hard_labels[idx], dtype=torch.long),
            'soft_label': torch.tensor(self.soft_labels[idx], dtype=torch.float)
        }


def map_9way_to_3way(probs_9):
    """Map 9-way probability distribution to 3-way by summing children."""
    probs_3 = np.zeros(3)
    for i, prob in enumerate(probs_9):
        probs_3[NINE_TO_THREE_MAPPING[i]] += prob
    return probs_3


def load_soft_labels_task1(filepath, num_samples):
    """Load 9-way soft labels and convert to 3-way."""
    soft_labels_3way = np.zeros((num_samples, 3))
    soft_labels_3way.fill(1.0 / 3.0)  # Uniform fallback
    
    valid_count = 0
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            idx = data['id']
            if idx < num_samples and data.get('valid', False):
                probs_9 = data['probs']
                if len(probs_9) == 9:
                    probs_3 = map_9way_to_3way(probs_9)
                    total = sum(probs_3)
                    if total > 0:
                        soft_labels_3way[idx] = probs_3 / total
                        valid_count += 1
    
    print(f"Loaded and mapped {valid_count} valid soft labels to 3-way")
    return soft_labels_3way


def train_task1_fold(fold, model_name, soft_labels_file, alpha, temperature,
                     batch_size=4, grad_accum=8, epochs=6, lr=2e-5):
    """Train one fold for Task 1."""
    
    print(f"\n{'='*60}")
    print(f"TASK 1: DEBERTA-XLARGE TRAINING - FOLD {fold}")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Alpha: {alpha}, Temp: {temperature}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    dataset = load_dataset("ailsntua/QEvasion")
    train_data = dataset['train']
    
    # Load splits
    project_root = Path(__file__).parent.parent.parent
    with open(project_root / "artifacts/splits/grouped_5fold_by_url.json") as f:
        fold_data = json.load(f)
    
    train_indices = fold_data['splits'][fold]['train_indices']
    val_indices = fold_data['splits'][fold]['val_indices']
    
    # Load 9-way soft labels and convert to 3-way
    soft_labels = load_soft_labels_task1(project_root / soft_labels_file, len(train_data))
    
    # Prepare texts and 3-way labels
    all_texts = [f"Question: {train_data[i]['question']}\nAnswer: {train_data[i]['interview_answer']}" 
                 for i in range(len(train_data))]
    all_labels = [TASK1_LABEL2ID[TASK2_TO_TASK1[train_data[i]['evasion_label']]] 
                  for i in range(len(train_data))]
    
    train_texts = [all_texts[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    train_soft = soft_labels[train_indices]
    
    val_texts = [all_texts[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Task1Classifier(model_name=model_name, num_labels=3)
    model = model.to(device)
    
    # Datasets
    train_dataset = Task1Dataset(train_texts, train_labels, train_soft, tokenizer)
    val_dataset = Task1Dataset(val_texts, val_labels, np.zeros((len(val_texts), 3)), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=2)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training
    best_f1 = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = project_root / f"runs/task1_xlarge_fold{fold}_{timestamp}"
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            hard_labels = batch['hard_label'].to(device)
            soft_labels_batch = batch['soft_label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            
            # Hard label CE loss
            ce_loss = F.cross_entropy(logits, hard_labels)
            
            # Soft label KL loss (3-way)
            student_log_probs = F.log_softmax(logits / temperature, dim=-1)
            teacher_probs = F.softmax(soft_labels_batch / temperature, dim=-1)
            kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
            
            # Combined loss
            loss = alpha * ce_loss + (1 - alpha) * kl_loss
            loss = loss / grad_accum
            loss.backward()
            
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
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
                labels = batch['hard_label']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs['logits']
                preds = logits.argmax(dim=-1).cpu()
                
                all_preds.extend(preds.tolist())
                all_true.extend(labels.tolist())
                all_logits.append(logits.cpu().numpy())
        
        # Calculate macro F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_true, all_preds, average='macro')
        
        print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}, Val Macro-F1={val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'fold': fold,
                'epoch': epoch,
                'val_f1': val_f1,
                'config': {
                    'alpha': alpha,
                    'temperature': temperature,
                    'model_name': model_name,
                    'task': 'task1'
                }
            }, save_dir / 'best_model.pt')
            
            # Save OOF logits for threshold tuning
            oof_logits = np.vstack(all_logits)
            np.save(save_dir / 'oof_logits.npy', oof_logits)
            np.save(save_dir / 'oof_labels.npy', np.array(all_true))
            
            print(f"[OK] New best model saved!")
    
    print(f"\nBest Val Macro-F1: {best_f1:.4f}")
    print(f"Saved to: {save_dir}")
    
    return best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-xlarge")
    parser.add_argument("--soft_labels_file", type=str, default="data/teacher/qwen7b_softlabels_clean.jsonl")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=6)
    args = parser.parse_args()
    
    train_task1_fold(
        fold=args.fold,
        model_name=args.model_name,
        soft_labels_file=args.soft_labels_file,
        alpha=args.alpha,
        temperature=args.temperature,
        batch_size=args.batch_size,
        grad_accum=args.gradient_accumulation_steps,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
