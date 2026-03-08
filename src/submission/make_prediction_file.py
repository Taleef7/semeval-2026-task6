"""
Generate CodaBench prediction files for Task 1 and Task 2.

Task 1: 3-way clarity classification
Task 2: 9-way evasion classification

Format requirements:
- Exactly 308 lines (one per test instance)
- One label per line
- Exact label strings (no extra whitespace)
- No headers, no indices
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.encoder_classifier import EvasionClassifier
from data.preprocess import prepare_model_inputs

# Label mappings
EVASION_LABELS = [
    "Explicit",
    "Implicit", 
    "Dodging",
    "General",
    "Deflection",
    "Partial/half-answer",
    "Declining to answer",
    "Claims ignorance",
    "Clarification"
]

CLARITY_LABELS = [
    "Clear Reply",
    "Ambivalent",
    "Clear Non-Reply"
]

# Taxonomy mapping: evasion -> clarity
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply"
}


def load_test_data(dataset_name: str = "ailsntua/QEvasion"):
    """Load test split from HuggingFace."""
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    test_data = dataset["test"]
    print(f"Test size: {len(test_data)}")
    
    if len(test_data) != 308:
        raise ValueError(f"Expected 308 test instances, got {len(test_data)}")
    
    return test_data


def load_model_checkpoint(checkpoint_path: Path, device: str = "cuda"):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Load checkpoint (PyTorch 2.6 compatibility)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract config
    model_name = checkpoint.get("model_name", "microsoft/deberta-v3-base")
    num_labels = checkpoint.get("num_labels", 9)
    
    print(f"Model: {model_name}, Classes: {num_labels}")
    
    # Initialize model
    model = EvasionClassifier(model_name=model_name, num_labels=num_labels)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    return model, model_name


def run_inference(
    model,
    tokenizer,
    test_data,
    max_length: int = 512,
    batch_size: int = 32,
    device: str = "cuda"
) -> List[int]:
    """Run inference on test data."""
    print(f"Running inference (batch_size={batch_size}, max_length={max_length})...")
    
    # Preprocess: create model_input column
    test_processed = prepare_model_inputs(test_data)
    
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(test_processed), batch_size):
            batch_end = min(i + batch_size, len(test_processed))
            batch = test_processed[i:batch_end]
            texts = batch["model_input"]
            
            # Tokenize
            encodings = tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encodings["input_ids"].to(device)
            attention_mask = encodings["attention_mask"].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Predictions
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_predictions.extend(preds)
            
            if (i // batch_size + 1) % 5 == 0:
                print(f"  Processed {batch_end}/{len(test_processed)}")
    
    print(f"[OK] Generated {len(all_predictions)} predictions")
    return all_predictions


def write_prediction_file(
    predictions: List[int],
    label_names: List[str],
    output_path: Path
):
    """Write predictions to file in CodaBench format."""
    print(f"Writing predictions to: {output_path}")
    
    # Convert indices to label strings
    pred_labels = [label_names[pred_idx] for pred_idx in predictions]
    
    # Validate
    if len(pred_labels) != 308:
        raise ValueError(f"Expected 308 predictions, got {len(pred_labels)}")
    
    # Write file (no header, one label per line, no extra whitespace)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for label in pred_labels:
            f.write(f"{label}\n")
    
    print(f"[OK] Wrote {len(pred_labels)} predictions")
    
    # Validate file
    with open(output_path, "r") as f:
        lines = f.readlines()
    
    if len(lines) != 308:
        raise ValueError(f"File has {len(lines)} lines, expected 308")
    
    print("[OK] Validation passed: 308 lines")


def derive_task1_from_task2(task2_predictions: List[int]) -> List[int]:
    """Derive Task 1 (clarity) predictions from Task 2 (evasion) predictions."""
    print("Deriving Task 1 predictions from Task 2 via taxonomy mapping...")
    
    task1_predictions = []
    for evasion_idx in task2_predictions:
        evasion_label = EVASION_LABELS[evasion_idx]
        clarity_label = EVASION_TO_CLARITY[evasion_label]
        clarity_idx = CLARITY_LABELS.index(clarity_label)
        task1_predictions.append(clarity_idx)
    
    # Print distribution
    task1_counts = {label: 0 for label in CLARITY_LABELS}
    for idx in task1_predictions:
        task1_counts[CLARITY_LABELS[idx]] += 1
    
    print("Task 1 prediction distribution:")
    for label, count in task1_counts.items():
        print(f"  {label}: {count} ({count/len(task1_predictions)*100:.1f}%)")
    
    return task1_predictions


def main():
    parser = argparse.ArgumentParser(description="Generate CodaBench prediction files")
    parser.add_argument("--checkpoint", type=str, required=True,
                      help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for predictions")
    parser.add_argument("--task", type=str, choices=["task1", "task2", "both"],
                      default="both", help="Which task(s) to generate predictions for")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    print("="*60)
    print("CODABENCH PREDICTION FILE GENERATION")
    print("="*60)
    
    # Load test data
    test_data = load_test_data()
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, model_name = load_model_checkpoint(checkpoint_path, args.device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Run inference (Task 2: evasion classification)
    task2_predictions = run_inference(
        model, tokenizer, test_data,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Print Task 2 distribution
    task2_counts = {label: 0 for label in EVASION_LABELS}
    for pred_idx in task2_predictions:
        task2_counts[EVASION_LABELS[pred_idx]] += 1
    
    print("\nTask 2 prediction distribution:")
    for label, count in task2_counts.items():
        print(f"  {label}: {count} ({count/len(task2_predictions)*100:.1f}%)")
    
    # Write Task 2 predictions
    if args.task in ["task2", "both"]:
        task2_output = Path(args.output_dir) / "task2" / "prediction"
        write_prediction_file(task2_predictions, EVASION_LABELS, task2_output)
        print(f"[OK] Task 2 predictions saved to: {task2_output}")
    
    # Derive and write Task 1 predictions
    if args.task in ["task1", "both"]:
        task1_predictions = derive_task1_from_task2(task2_predictions)
        task1_output = Path(args.output_dir) / "task1" / "prediction"
        write_prediction_file(task1_predictions, CLARITY_LABELS, task1_output)
        print(f"[OK] Task 1 predictions saved to: {task1_output}")
    
    print("\n" + "="*60)
    print("[OK] PREDICTION FILE GENERATION COMPLETE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Zip the prediction files using zip_submission.py")
    print(f"  2. Upload to CodaBench:")
    print(f"     - Task 1: https://www.codabench.org/competitions/10879/")
    print(f"     - Task 2: https://www.codabench.org/competitions/11131/")


if __name__ == "__main__":
    main()
