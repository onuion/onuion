"""
JSON to NPZ Converter

Converts JSON session files to .npz format for model training.
Reads JSON files from a directory and converts them to feature vectors.
"""

import json
import os
import sys
import numpy as np
import argparse
from glob import glob
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onuion.feature_extractor import FeatureExtractor


def load_json_sessions(json_dir: str, label: float = None) -> Tuple[List[dict], List[float]]:
    """
    Loads all JSON session files from a directory.
    
    Args:
        json_dir: Directory containing JSON files
        label: Label for all sessions (0.0 for low risk, 1.0 for high risk)
        
    Returns:
        (sessions, labels) tuple
    """
    # Use pathlib for better cross-platform support
    json_dir_path = Path(json_dir).resolve()
    if not json_dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {json_dir}")
    
    json_files = list(json_dir_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {json_dir}")
    
    # Convert Path objects to strings
    json_files = [str(f) for f in json_files]
    
    sessions = []
    labels = []
    
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            session = json.load(f)
            sessions.append(session)
            if label is not None:
                labels.append(label)
    
    return sessions, labels


def convert_json_to_npz(
    low_risk_dir: str = None,
    medium_risk_dir: str = None,
    high_risk_dir: str = None,
    json_dir: str = None,
    output_path: str = "data/training_data.npz",
    train_split: float = 0.8
):
    """
    Converts JSON session files to .npz format for training.
    
    Args:
        low_risk_dir: Directory with low-risk JSON files (label: 0.0)
        medium_risk_dir: Directory with medium-risk JSON files (label: 0.5)
        high_risk_dir: Directory with high-risk JSON files (label: 1.0)
        json_dir: Single directory with all JSON files (auto-detect label from filename)
        output_path: Output .npz file path
        train_split: Train/validation split ratio
    """
    extractor = FeatureExtractor()
    all_features = []
    all_labels = []
    
    # Load sessions from separate directories
    if low_risk_dir:
        print(f"Loading low-risk sessions from: {low_risk_dir}")
        sessions, labels = load_json_sessions(low_risk_dir, label=0.0)
        for session in sessions:
            features = extractor.extract(session)
            all_features.append(features)
            all_labels.append(0.0)
        print(f"  Loaded {len(sessions)} low-risk sessions")
    
    if medium_risk_dir:
        print(f"Loading medium-risk sessions from: {medium_risk_dir}")
        sessions, labels = load_json_sessions(medium_risk_dir, label=0.5)
        for session in sessions:
            features = extractor.extract(session)
            all_features.append(features)
            all_labels.append(0.5)
        print(f"  Loaded {len(sessions)} medium-risk sessions")
    
    if high_risk_dir:
        print(f"Loading high-risk sessions from: {high_risk_dir}")
        sessions, labels = load_json_sessions(high_risk_dir, label=1.0)
        for session in sessions:
            features = extractor.extract(session)
            all_features.append(features)
            all_labels.append(1.0)
        print(f"  Loaded {len(sessions)} high-risk sessions")
    
    # Load from single directory (auto-detect label from filename)
    if json_dir:
        print(f"Loading sessions from: {json_dir}")
        # Normalize path and check if directory exists
        json_dir_path = Path(json_dir).resolve()
        if not json_dir_path.is_dir():
            raise ValueError(f"Directory does not exist: {json_dir}")
        
        # Find all JSON files using pathlib (more reliable on Windows)
        json_files = list(json_dir_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {json_dir}")
        
        # Convert Path objects to strings
        json_files = [str(f) for f in json_files]
        
        print(f"  Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                session = json.load(f)
                features = extractor.extract(session)
                all_features.append(features)
                
                # Auto-detect label from filename
                filename = os.path.basename(json_file)
                if "low_risk" in filename:
                    all_labels.append(0.0)
                elif "medium_risk" in filename:
                    all_labels.append(0.5)
                elif "high_risk" in filename:
                    all_labels.append(1.0)
                else:
                    all_labels.append(0.0)  # Default to low risk
        
        print(f"  Loaded {len(json_files)} sessions")
    
    if not all_features:
        raise ValueError("No sessions loaded! Please specify at least one directory.")
    
    # Convert to numpy arrays
    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.float32)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Train/validation split
    split_idx = int(len(X) * train_split)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    # Convert binary labels (0.0 or 1.0) for binary classification
    # Medium risk (0.5) can be treated as 1.0 for binary classification
    y_train_binary = (y_train >= 0.5).astype(np.float32)
    y_val_binary = (y_val >= 0.5).astype(np.float32)
    
    # Save to .npz file
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train_binary,
        X_val=X_val,
        y_val=y_val_binary
    )
    
    print(f"\n[OK] Conversion complete!")
    print(f"  Total samples: {len(X)}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Features per sample: {X_train.shape[1]}")
    print(f"  Saved to: {output_path}")
    print(f"\n  Label distribution:")
    print(f"    Low risk (0.0): {np.sum(y_train_binary == 0.0)} train, {np.sum(y_val_binary == 0.0)} val")
    print(f"    High risk (1.0): {np.sum(y_train_binary == 1.0)} train, {np.sum(y_val_binary == 1.0)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON session files to .npz format")
    parser.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help="Directory containing all JSON files (auto-detect label from filename)"
    )
    parser.add_argument(
        "--low-risk-dir",
        type=str,
        default=None,
        help="Directory with low-risk JSON files"
    )
    parser.add_argument(
        "--medium-risk-dir",
        type=str,
        default=None,
        help="Directory with medium-risk JSON files"
    )
    parser.add_argument(
        "--high-risk-dir",
        type=str,
        default=None,
        help="Directory with high-risk JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training_data.npz",
        help="Output .npz file path"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    convert_json_to_npz(
        low_risk_dir=args.low_risk_dir,
        medium_risk_dir=args.medium_risk_dir,
        high_risk_dir=args.high_risk_dir,
        json_dir=args.json_dir,
        output_path=args.output,
        train_split=args.train_split
    )

