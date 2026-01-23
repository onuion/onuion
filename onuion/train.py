"""
Training Script

Used to train the model.
Can train with synthetic data or real data.
"""

import numpy as np
import argparse
import os
from typing import Tuple

from onuion.model import RiskModel
from onuion.feature_extractor import FeatureExtractor


def generate_synthetic_data(
    n_samples: int = 10000, n_features: int = 25, noise_level: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic training data.

    Real labeled data should be used in production!
    This function is only for testing and development.

    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise_level: Noise level

    Returns:
        (X, y) tuple
    """
    np.random.seed(42)

    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)

    # Calculate risk score (simple rule-based)
    # High IP change, geo difference, request rate = risk
    risk_scores = (
        np.abs(X[:, 0]) * 0.3  # IP change
        + np.abs(X[:, 4]) * 0.3  # Geo difference
        + np.abs(X[:, 12]) * 0.2  # Request rate
        + np.abs(X[:, 15]) * 0.2  # Suspicious pattern
    )

    # Normalize (0-1 range)
    risk_scores = (risk_scores - risk_scores.min()) / (risk_scores.max() - risk_scores.min() + 1e-8)

    # Add noise
    risk_scores += np.random.normal(0, noise_level, n_samples)
    risk_scores = np.clip(risk_scores, 0, 1)

    # Binary label (threshold: 0.5)
    y = (risk_scores > 0.5).astype(np.float32)

    return X, y


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 50,
    batch_size: int = 32,
    output_dir: str = "models/onuion_model",
):
    """
    Trains and saves the model.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        epochs: Number of epochs
        batch_size: Batch size
        output_dir: Model save directory
    """
    # Create model
    input_dim = X_train.shape[1]
    model = RiskModel(input_dim=input_dim)

    print("Model architecture:")
    print(model.get_model_summary())
    print(f"\nTotal parameter count: {model.get_parameter_count():,}")

    # Training
    print("\nTraining starting...")
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)

    print(f"\nModel saved: {output_dir}")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    if X_val is not None:
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")

    return model, history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="onuion Model Training")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data (for testing)")
    parser.add_argument(
        "--data-path", type=str, default=None, help="Real data path (numpy .npz format)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/onuion_model", help="Model save directory"
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    # Data loading
    if args.synthetic:
        print("Generating synthetic data...")
        X, y = generate_synthetic_data(n_samples=10000)

        # Train/Val split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

    elif args.data_path:
        print(f"Loading data: {args.data_path}")
        data = np.load(args.data_path)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_val = data.get("X_val", None)
        y_val = data.get("y_val", None)

        print(f"Training samples: {len(X_train)}")
        if X_val is not None:
            print(f"Validation samples: {len(X_val)}")

    else:
        print("ERROR: --synthetic or --data-path must be specified!")
        return

    # Model training
    train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
