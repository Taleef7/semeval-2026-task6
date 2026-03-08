import argparse
import json
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import gc

# Add src to path for importing our model
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.encoder_classifier import EvasionClassifier

# --- Helper: deterministic label mapping (from preprocess.py) ---
EVASION_LABELS = [
    "Explicit", "Implicit", "Dodging", "General", "Deflection", "Partial/half-answer",
    "Declining to answer", "Claims ignorance", "Clarification"
]
EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "General": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}
CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]


def parse_args():
    parser = argparse.ArgumentParser(description="Ensemble prediction file generator for CLARITY Task 2/1.")
    parser.add_argument('--manifest', type=str, required=True, help='Path to ensemble manifest JSON')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--save_logits', action='store_true', help='Save averaged logits/probs for inspection')
    parser.add_argument('--max_examples', type=int, default=None, help='If set, only run on this many test examples (for sanity check)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference (default: 32)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto, cuda, cpu)')
    return parser.parse_args()


def create_input_text(example):
    """Create model input text from example."""
    interview_q = example.get("interview_question", "").strip()
    question = example.get("question", "").strip()
    answer = example.get("interview_answer", "").strip()
    return (
        f"Interview question (context): {interview_q}\n"
        f"Target sub-question: {question}\n"
        f"Answer: {answer}"
    )


def main():
    args = parse_args()
    
    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: Manifest file not found: {args.manifest}")
        sys.exit(1)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    checkpoints = manifest['checkpoints']
    model_name = manifest['model_name']
    max_length = manifest.get('max_length', 512)
    
    # Validate checkpoints exist
    print(f"[INFO] Ensemble configuration:")
    print(f"  Model: {model_name}")
    print(f"  Max length: {max_length}")
    print(f"  Number of checkpoints: {len(checkpoints)}")
    for i, ckpt in enumerate(checkpoints):
        ckpt_path = Path(ckpt)
        if not ckpt_path.exists():
            print(f"ERROR: Checkpoint {i+1} not found: {ckpt}")
            sys.exit(1)
        print(f"  [{i+1}] {ckpt_path.name}")

    # Load test set (preserve order)
    print(f"\n[INFO] Loading test dataset...")
    ds = load_dataset("ailsntua/QEvasion", split="test")
    if args.max_examples:
        ds = ds.select(range(args.max_examples))
        print(f"  Limited to {args.max_examples} examples for testing")
    
    num_examples = len(ds)
    print(f"  Loaded {num_examples} test examples")
    
    if num_examples != 308:
        print(f"  WARNING: Expected 308 examples, got {num_examples}")

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"\n[INFO] Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    all_logits = []

    # Process each checkpoint
    for ckpt_idx, ckpt in enumerate(checkpoints, 1):
        ckpt_name = Path(ckpt).name
        print(f"\n[INFO] Processing checkpoint {ckpt_idx}/{len(checkpoints)}: {ckpt_name}")
        
        # Load model using our EvasionClassifier (same as training)
        try:
            print(f"  Loading checkpoint...")
            model = EvasionClassifier(model_name=model_name, num_labels=9)
            
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Load state dict directly (EvasionClassifier structure)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print(f"  Model loaded successfully")
        except Exception as e:
            print(f"  ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
        # Batch inference for efficiency
        logits = []
        batch_size = args.batch_size
        
        print(f"  Running inference (batch_size={batch_size})...")
        for i in tqdm(range(0, num_examples, batch_size), desc=f"  Processing batches"):
            batch_examples = ds.select(range(i, min(i + batch_size, num_examples)))
            
            # Prepare batch texts
            batch_texts = [create_input_text(ex) for ex in batch_examples]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors='pt'
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                batch_logits = outputs["logits"].cpu().numpy()
                logits.append(batch_logits)
        
        # Stack all logits for this model
        model_logits = np.vstack(logits)
        all_logits.append(model_logits)
        print(f"  [OK] Completed: {model_logits.shape}")
        # Cleanup
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Ensemble averaging
    print(f"\n[INFO] Computing ensemble predictions...")
    all_logits = np.stack(all_logits)  # [n_models, N, 9]
    print(f"  Logits shape: {all_logits.shape}")
    
    avg_logits = np.mean(all_logits, axis=0)  # [N, 9]
    y_pred = np.argmax(avg_logits, axis=1)
    
    # Validate predictions
    if len(y_pred) != num_examples:
        print(f"ERROR: Prediction count mismatch: {len(y_pred)} != {num_examples}")
        sys.exit(1)
    
    # Check label validity
    invalid_labels = [i for i, pred in enumerate(y_pred) if pred < 0 or pred >= len(EVASION_LABELS)]
    if invalid_labels:
        print(f"ERROR: Invalid predictions at indices: {invalid_labels[:10]}")
        sys.exit(1)
    
    print(f"  [OK] Ensemble predictions computed: {len(y_pred)} predictions")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write Task 2 predictions
    task2_path = output_dir / 'task2_prediction'
    print(f"\n[INFO] Writing Task 2 predictions to {task2_path}...")
    with open(task2_path, 'w') as f:
        for pred_idx in y_pred:
            f.write(EVASION_LABELS[pred_idx] + '\n')
    
    # Validate Task 2 file
    with open(task2_path) as f:
        task2_lines = f.readlines()
    if len(task2_lines) != num_examples:
        print(f"ERROR: Task 2 file has {len(task2_lines)} lines, expected {num_examples}")
        sys.exit(1)
    print(f"  [OK] Task 2: {len(task2_lines)} lines written")

    # Derive Task 1 predictions
    task1_path = output_dir / 'task1_prediction'
    print(f"\n[INFO] Writing Task 1 predictions to {task1_path}...")
    with open(task1_path, 'w') as f:
        for pred_idx in y_pred:
            evasion = EVASION_LABELS[pred_idx]
            clarity = EVASION_TO_CLARITY[evasion]
            f.write(clarity + '\n')
    
    # Validate Task 1 file
    with open(task1_path) as f:
        task1_lines = f.readlines()
    if len(task1_lines) != num_examples:
        print(f"ERROR: Task 1 file has {len(task1_lines)} lines, expected {num_examples}")
        sys.exit(1)
    print(f"  [OK] Task 1: {len(task1_lines)} lines written")

    # Save logits/probs for inspection
    if args.save_logits:
        logits_path = output_dir / 'avg_logits.npy'
        preds_path = output_dir / 'y_pred.npy'
        np.save(logits_path, avg_logits)
        np.save(preds_path, y_pred)
        print(f"  [OK] Logits saved: {logits_path}")
        print(f"  [OK] Predictions saved: {preds_path}")

    # Print prediction distribution
    print(f"\n[INFO] Prediction distribution (Task 2):")
    unique, counts = np.unique(y_pred, return_counts=True)
    for label_idx, count in zip(unique, counts):
        pct = 100 * count / num_examples
        print(f"  {EVASION_LABELS[label_idx]:25s}: {count:3d} ({pct:5.2f}%)")

    print(f"\n{'='*60}")
    print(f"[OK] Ensemble inference completed successfully")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"  Task 2: {task2_path}")
    print(f"  Task 1: {task1_path}")
    if args.save_logits:
        print(f"  Logits: {logits_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
