"""
Create CodaBench submission zip files.

Format requirements:
- Zip file contains exactly one file named 'prediction' (no extension)
- No subdirectories in the zip
"""

import argparse
import zipfile
from pathlib import Path
from datetime import datetime


def validate_prediction_file(pred_file: Path):
    """Validate prediction file format."""
    print(f"Validating: {pred_file}")
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    
    if pred_file.name != "prediction":
        raise ValueError(f"File must be named 'prediction' (no extension), got: {pred_file.name}")
    
    # Check line count
    with open(pred_file, "r") as f:
        lines = f.readlines()
    
    if len(lines) != 308:
        raise ValueError(f"Expected 308 lines, got {len(lines)}")
    
    # Check for extra whitespace
    for i, line in enumerate(lines):
        stripped = line.rstrip("\n")
        if line != stripped + "\n":
            raise ValueError(f"Line {i+1} has extra whitespace: {repr(line)}")
    
    print(f"  [OK] 308 lines")
    print(f"  [OK] No extra whitespace")
    print(f"  [OK] Correct filename")


def create_submission_zip(pred_file: Path, output_zip: Path):
    """Create submission zip."""
    print(f"Creating zip: {output_zip}")
    
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add file at root level (no subdirectories)
        zf.write(pred_file, arcname="prediction")
    
    # Verify zip contents
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        if names != ["prediction"]:
            raise ValueError(f"Zip should contain only 'prediction', got: {names}")
    
    print(f"  [OK] Zip created")
    print(f"  [OK] Contains: {names}")
    print(f"  [OK] Size: {output_zip.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Create CodaBench submission zip")
    parser.add_argument("--prediction_file", type=str, required=True,
                      help="Path to prediction file")
    parser.add_argument("--output_zip", type=str, required=True,
                      help="Output zip file path")
    parser.add_argument("--skip_validation", action="store_true",
                      help="Skip validation checks")
    
    args = parser.parse_args()
    
    pred_file = Path(args.prediction_file)
    output_zip = Path(args.output_zip)
    
    print("="*60)
    print("CODABENCH SUBMISSION ZIP CREATION")
    print("="*60)
    
    # Validate
    if not args.skip_validation:
        validate_prediction_file(pred_file)
    
    # Create zip
    create_submission_zip(pred_file, output_zip)
    
    print("\n" + "="*60)
    print("[OK] SUBMISSION ZIP CREATED")
    print("="*60)
    print(f"\nSubmission ready: {output_zip}")
    print(f"\nUpload to CodaBench:")
    if "task1" in str(output_zip):
        print(f"  https://www.codabench.org/competitions/10879/")
    elif "task2" in str(output_zip):
        print(f"  https://www.codabench.org/competitions/11131/")


if __name__ == "__main__":
    main()
