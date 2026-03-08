#!/usr/bin/env python3
"""
Generate final optimized ensemble predictions using:
1. Top 70% of v3-large models (by OOF F1)
2. All xlarge models
3. OOF-weighted averaging between architectures
4. Hierarchical masking for Task 2
"""

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, DebertaV2Tokenizer
import zipfile
from datetime import datetime

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


class TaskClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels):
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


def apply_hierarchical_masking(task2_logits, task1_pred):
    """Mask Task 2 logits based on Task 1 predictions."""
    masked_logits = task2_logits.copy()
    
    # Define masking rules based on Task 1 clarity
    if task1_pred == "Clear Reply":
        # Boost explicit/direct answers, suppress evasive types
        boost_indices = [5]  # Explicit
        suppress_indices = [0, 1, 2, 3, 4, 6, 7]  # Evasive types
        for idx in suppress_indices:
            masked_logits[idx] *= 0.3
            
    elif task1_pred == "Clear Non-Reply":
        # Boost evasive types, suppress explicit
        suppress_indices = [5]  # Explicit
        for idx in suppress_indices:
            masked_logits[idx] *= 0.1
            
    # Ambivalent: no strong masking
    
    return masked_logits


def generate_predictions(task_num, eval_csv_path, use_hierarchical=False, task1_preds=None):
    """Generate ensemble predictions for eval set."""
    
    print(f"\n{'='*70}")
    print(f"GENERATING TASK {task_num} PREDICTIONS")
    print(f"{'='*70}")
    
    project_root = Path.cwd()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load evaluation data
    import pandas as pd
    eval_df = pd.read_csv(eval_csv_path)
    eval_texts = [f"Question: {row['question']}\nAnswer: {row['interview_answer']}" 
                  for _, row in eval_df.iterrows()]
    
    n_samples = len(eval_texts)
    label_list = TASK1_LABELS if task_num == 1 else EVASION_LABELS
    n_classes = len(label_list)
    
    print(f"Evaluation samples: {n_samples}")
    print(f"Classes: {n_classes}")
    
    # Load selection info
    with open(project_root / "artifacts/oof/ensemble_selection.json") as f:
        selection = json.load(f)
    
    # Load model scores to select top 70%
    with open(project_root / f"artifacts/oof/task{task_num}_v3large_model_scores.json") as f:
        v3_scores = json.load(f)
    
    top_70_threshold = selection[f'task{task_num}']['top_70_percent_threshold'] if task_num == 1 else 0.0
    selected_v3_models = [m for m in v3_scores['models'] if m['f1'] >= top_70_threshold] if task_num == 1 else v3_scores['models']
    
    print(f"\nModel selection:")
    print(f"  v3-large: {len(selected_v3_models)}/{len(v3_scores['models'])} models")
    
    # Initialize ensemble logits
    v3_ensemble_logits = np.zeros((n_samples, n_classes))
    xl_ensemble_logits = np.zeros((n_samples, n_classes))
    
    # === Process v3-large models ===
    print(f"\n{'='*70}")
    print("LOADING V3-LARGE MODELS")
    print(f"{'='*70}")
    
    v3_model_name = "microsoft/deberta-v3-large"
    v3_tokenizer = DebertaV2Tokenizer.from_pretrained(v3_model_name)
    batch_size = 8
    
    for model_info in tqdm(selected_v3_models, desc="v3-large models"):
        model_dir = Path(model_info['dir'])
        
        # Load model with correct architecture
        model = TaskClassifier(v3_model_name, num_labels=n_classes)
        checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Predict
        batch_logits = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_texts = eval_texts[i:i+batch_size]
                encoding = v3_tokenizer(
                    batch_texts,
                    max_length=512,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                batch_logits.append(outputs['logits'].cpu().numpy())
        
        logits = np.vstack(batch_logits)
        v3_ensemble_logits += logits
        
        del model, checkpoint
        torch.cuda.empty_cache()
    
    # Average v3-large predictions
    v3_ensemble_logits /= len(selected_v3_models)
    
    # === Process xlarge models (Task 1 only for now) ===
    if task_num == 1:
        print(f"\n{'='*70}")
        print("LOADING XLARGE MODELS")
        print(f"{'='*70}")
        
        # Find xlarge checkpoints
        xlarge_dirs = sorted(project_root.glob("runs/task1_xlarge_*"))
        
        # Filter to latest per fold/seed
        from collections import defaultdict
        latest_dirs = defaultdict(lambda: (None, ""))
        for model_dir in xlarge_dirs:
            parts = model_dir.name.split("_")
            if len(parts) >= 4:
                fold = parts[2] if len(parts) > 2 else ""
                seed = parts[3] if len(parts) > 3 else ""
                timestamp = "_".join(parts[4:]) if len(parts) > 4 else ""
                key = (fold, seed)
                if timestamp > latest_dirs[key][1]:
                    latest_dirs[key] = (model_dir, timestamp)
        
        xlarge_model_dirs = sorted([d[0] for d in latest_dirs.values() if d[0] is not None])
        print(f"Found {len(xlarge_model_dirs)} xlarge models")
        
        xl_model_name = "microsoft/deberta-v2-xlarge"
        xl_tokenizer = DebertaV2Tokenizer.from_pretrained(xl_model_name)
        
        for model_dir in tqdm(xlarge_model_dirs[:50], desc="xlarge models"):  # Limit to 50
            if not (model_dir / "best_model.pt").exists():
                continue
                
            model = TaskClassifier(xl_model_name, num_labels=n_classes)
            checkpoint = torch.load(model_dir / "best_model.pt", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            batch_logits = []
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    batch_texts = eval_texts[i:i+batch_size]
                    encoding = xl_tokenizer(
                        batch_texts,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    )
                    input_ids = encoding['input_ids'].to(device)
                    attention_mask = encoding['attention_mask'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    batch_logits.append(outputs['logits'].cpu().numpy())
            
            logits = np.vstack(batch_logits)
            xl_ensemble_logits += logits
            
            del model, checkpoint
            torch.cuda.empty_cache()
        
        xl_ensemble_logits /= len(xlarge_model_dirs[:50])
    
    # === Combine architectures with OOF weights ===
    print(f"\n{'='*70}")
    print("COMBINING ARCHITECTURES")
    print(f"{'='*70}")
    
    if task_num == 1:
        v3_weight = selection['task1']['v3_weight']
        xl_weight = selection['task1']['xl_weight']
        print(f"v3-large weight: {v3_weight:.3f}")
        print(f"xlarge weight: {xl_weight:.3f}")
        final_logits = v3_weight * v3_ensemble_logits + xl_weight * xl_ensemble_logits
    else:
        # Task 2: only v3-large for now
        final_logits = v3_ensemble_logits
    
    # === Apply hierarchical masking for Task 2 ===
    if use_hierarchical and task1_preds is not None:
        print(f"\n{'='*70}")
        print("APPLYING HIERARCHICAL MASKING")
        print(f"{'='*70}")
        
        masked_count = 0
        for i in range(n_samples):
            original_pred = final_logits[i].argmax()
            masked_logits = apply_hierarchical_masking(final_logits[i], task1_preds[i])
            final_logits[i] = masked_logits
            
            if masked_logits.argmax() != original_pred:
                masked_count += 1
        
        print(f"Predictions changed by masking: {masked_count}/{n_samples} ({100*masked_count/n_samples:.1f}%)")
    
    # === Generate final predictions ===
    predictions = [label_list[idx] for idx in final_logits.argmax(axis=1)]
    
    return predictions, final_logits


def main():
    project_root = Path.cwd()
    eval_csv = project_root / "clarity_task_evaluation_dataset.csv"
    
    # Generate Task 1 predictions
    task1_preds, task1_logits = generate_predictions(1, eval_csv, use_hierarchical=False)
    
    # Generate Task 2 predictions with hierarchical masking
    task2_preds, task2_logits = generate_predictions(2, eval_csv, use_hierarchical=True, task1_preds=task1_preds)
    
    # Save predictions
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / f"submissions/optimized_ensemble_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Task 1
    with open(output_dir / "task1_prediction", "w") as f:
        for pred in task1_preds:
            f.write(pred + "\n")
    
    # Task 2
    with open(output_dir / "task2_prediction", "w") as f:
        for pred in task2_preds:
            f.write(pred + "\n")
    
    # Create zip files
    for task in [1, 2]:
        zip_path = output_dir / f"task{task}_submission.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(output_dir / f"task{task}_prediction", arcname=f"task{task}_prediction")
        print(f"\nCreated {zip_path}")
    
    print(f"\n{'='*70}")
    print("SUBMISSION READY!")
    print(f"{'='*70}")
    print(f"Location: {output_dir}")
    print(f"Task 1: {output_dir}/task1_submission.zip")
    print(f"Task 2: {output_dir}/task2_submission.zip")


if __name__ == "__main__":
    main()
