#!/usr/bin/env python3
"""
Local Evaluation Script for Evaluation Phase

This allows us to test predictions locally without using Codabench submissions.
Requires the evaluation phase solution file (ground truth labels).
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import zipfile

TASK1_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
TASK2_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection",
    "Partial/half-answer", "Declining to answer", "Claims ignorance", "Clarification"
]


def load_predictions(pred_file):
    """Load predictions from file (either raw or from zip)."""
    if pred_file.endswith('.zip'):
        # Extract prediction file from zip
        with zipfile.ZipFile(pred_file, 'r') as zf:
            with zf.open('prediction') as f:
                return [line.decode('utf-8').strip() for line in f]
    else:
        # Read directly
        with open(pred_file) as f:
            return [line.strip() for line in f]


def load_ground_truth(solution_file, task='task1'):
    """
    Load ground truth from solution file.
    
    The solution file format depends on the competition setup.
    Common formats:
    1. CSV with columns: id, clarity_label, evasion_label
    2. Separate files: task1_solution.txt, task2_solution.txt
    3. JSON with structure: {"task1": [...], "task2": [...]}
    """
    solution_path = Path(solution_file)
    
    # Try different formats
    if solution_path.suffix == '.csv':
        df = pd.read_csv(solution_path)
        if task == 'task1':
            return df['clarity_label'].tolist()
        else:
            return df['evasion_label'].tolist()
    
    elif solution_path.suffix == '.json':
        with open(solution_path) as f:
            data = json.load(f)
        return data[task]
    
    elif solution_path.suffix == '.txt':
        with open(solution_path) as f:
            return [line.strip() for line in f]
    
    elif solution_path.suffix == '.zip':
        # Extract and try to find task-specific file
        with zipfile.ZipFile(solution_path, 'r') as zf:
            # List all files
            files = zf.namelist()
            print(f"Files in solution zip: {files}")
            
            # Look for task-specific file
            task_file = f"{task}_solution.txt" if f"{task}_solution.txt" in files else "solution.txt"
            if task_file in files:
                with zf.open(task_file) as f:
                    return [line.decode('utf-8').strip() for line in f]
    
    raise ValueError(f"Unknown solution file format: {solution_path}")


def evaluate_predictions(pred_file, solution_file, task='task1'):
    """Evaluate predictions against ground truth."""
    
    print("="*60)
    print(f"LOCAL EVALUATION: {task.upper()}")
    print("="*60)
    
    # Load data
    predictions = load_predictions(pred_file)
    ground_truth = load_ground_truth(solution_file, task)
    
    assert len(predictions) == len(ground_truth), \
        f"Mismatch: {len(predictions)} predictions vs {len(ground_truth)} ground truth"
    
    # Get label set
    labels = TASK1_LABELS if task == 'task1' else TASK2_LABELS
    
    # Compute macro F1
    macro_f1 = f1_score(ground_truth, predictions, labels=labels, average='macro', zero_division=0)
    
    print(f"\n" + "="*60)
    print(f"MACRO F1 SCORE: {macro_f1:.4f}")
    print("="*60)
    
    # Detailed report
    print("\nPer-class F1 scores:")
    print(classification_report(ground_truth, predictions, labels=labels, zero_division=0))
    
    # Confusion analysis
    print("\nPrediction distribution:")
    pred_dist = Counter(predictions)
    true_dist = Counter(ground_truth)
    
    for label in labels:
        pred_count = pred_dist.get(label, 0)
        true_count = true_dist.get(label, 0)
        delta = pred_count - true_count
        print(f"  {label:25s}: pred={pred_count:3d}, true={true_count:3d}, Δ={delta:+3d}")
    
    return macro_f1


def main():
    parser = argparse.ArgumentParser(description="Local evaluation for SemEval competition")
    parser.add_argument("--task1_pred", type=str, help="Task 1 prediction file (zip or txt)")
    parser.add_argument("--task2_pred", type=str, help="Task 2 prediction file (zip or txt)")
    parser.add_argument("--solution", type=str, required=True, help="Solution file with ground truth")
    args = parser.parse_args()
    
    results = {}
    
    if args.task1_pred:
        results['task1'] = evaluate_predictions(args.task1_pred, args.solution, task='task1')
    
    if args.task2_pred:
        results['task2'] = evaluate_predictions(args.task2_pred, args.solution, task='task2')
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task, score in results.items():
        print(f"{task.upper()}: {score:.4f}")
    
    if len(results) == 2:
        avg_score = np.mean(list(results.values()))
        print(f"\nAVERAGE: {avg_score:.4f}")


if __name__ == "__main__":
    main()
