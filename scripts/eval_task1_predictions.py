#!/usr/bin/env python3
"""
Generate Task 1 predictions on evaluation dataset using gold 0.64 model
"""

import os
import sys
import json
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


def main():
    print("="*60)
    print("TASK 1 EVALUATION PREDICTIONS (Gold 0.64 Model)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "microsoft/deberta-xlarge"
    
    # Load evaluation data
    project_root = Path(__file__).parent.parent
    eval_df = pd.read_csv(project_root / "clarity_task_evaluation_dataset.csv")
    
    print(f"\nEvaluation samples: {len(eval_df)}")
    
    # Prepare texts
    eval_texts = []
    for _, row in eval_df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['interview_answer']}"
        eval_texts.append(text)
    
    # Find gold model checkpoints (5 folds)
    gold_models = []
    for fold in range(5):
        # Match task1_xlarge_ce_fold* or task1_xlarge_fold* patterns
        model_dirs = sorted((project_root / "runs").glob(f"task1_xlarge*fold{fold}_*"))
        if len(model_dirs) > 0:
            gold_models.append(model_dirs[-1] / 'best_model.pt')
            print(f"Fold {fold}: {model_dirs[-1].name}")
    
    if len(gold_models) != 5:
        raise ValueError(f"Found {len(gold_models)} Task 1 checkpoints, expected 5")
    
    # Initialize ensemble
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eval_dataset = SimpleDataset(eval_texts, tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    ensemble_logits = np.zeros((len(eval_texts), len(TASK1_LABELS)))
    
    # Process each fold
    for fold, ckpt_path in enumerate(gold_models):
        print(f"\nProcessing Fold {fold}...")
        
        model = Task1Classifier(model_name=model_name, num_labels=3)
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
        
        fold_logits = np.vstack(batch_logits)
        ensemble_logits += fold_logits
        
        del model
        torch.cuda.empty_cache()
    
    # Average and predict
    ensemble_logits /= len(gold_models)
    predictions = [TASK1_LABELS[idx] for idx in ensemble_logits.argmax(axis=1)]
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / f"submissions/eval_task1_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "prediction", "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")
    
    # Create zip
    import shutil
    shutil.make_archive(str(output_dir / "task1_submission"), 'zip', output_dir, "prediction")
    
    from collections import Counter
    dist = Counter(predictions)
    
    print(f"\n{'='*60}")
    print("TASK 1 EVAL SUBMISSION CREATED")
    print(f"{'='*60}")
    print(f"Directory: {output_dir}")
    print(f"Zip: {output_dir}/task1_submission.zip")
    print(f"\nPrediction distribution:")
    for label in TASK1_LABELS:
        print(f"  {label:20s}: {dist.get(label, 0):4d} ({100*dist.get(label, 0)/len(predictions):.1f}%)")


if __name__ == "__main__":
    main()
