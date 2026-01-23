"""
Model Format Converter

Converts trained models to different formats:
- .keras (Keras 3 format)
- .h5 (H5 format)
- .tflite (TensorFlow Lite format)
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from onuion.model import RiskModel


def convert_model(
    model_path: str,
    output_format: str,
    output_path: str = None,
    quantization: str = "none",
    representative_data_path: str = None
):
    """
    Converts a model to the specified format.
    
    Args:
        model_path: Path to the input model (SavedModel directory or .keras file)
        output_format: Target format - "keras", "h5", or "tflite"
        output_path: Output file path (optional, auto-generated if not provided)
        quantization: Quantization mode for TFLite - "none", "int8", "float16", "dynamic_range"
        representative_data_path: Path to .npz file with representative data for int8 quantization
    """
    print(f"Loading model from: {model_path}")
    model = RiskModel()
    model.load(model_path)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path.rstrip('/\\')))[0]
        if output_format == "keras":
            output_path = f"models/{base_name}.keras"
        elif output_format == "h5":
            output_path = f"models/{base_name}.h5"
        elif output_format == "tflite":
            output_path = f"models/{base_name}.tflite"
        else:
            raise ValueError(f"Unknown format: {output_format}")
    
    # Load representative dataset if needed
    representative_dataset = None
    if quantization == "int8" and representative_data_path:
        print(f"Loading representative dataset from: {representative_data_path}")
        data = np.load(representative_data_path)
        # Use training data as representative dataset
        if "X_train" in data:
            representative_dataset = data["X_train"][:100]  # Use first 100 samples
            print(f"  Using {len(representative_dataset)} samples for quantization")
        else:
            raise ValueError("Representative dataset must contain 'X_train' key")
    
    # Convert based on format
    if output_format == "keras":
        output_path = model.convert_to_keras(output_path)
    elif output_format == "h5":
        output_path = model.convert_to_h5(output_path)
    elif output_format == "tflite":
        output_path = model.convert_to_tflite(
            output_path,
            quantization=quantization,
            representative_dataset=representative_dataset
        )
    else:
        raise ValueError(f"Unknown format: {output_format}")
    
    print(f"\n[OK] Conversion complete!")
    print(f"  Input: {model_path}")
    print(f"  Output: {output_path}")
    print(f"  Format: {output_format}")
    if output_format == "tflite":
        print(f"  Quantization: {quantization}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert onuion model to different formats")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to input model (SavedModel directory or .keras file)"
    )
    parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["keras", "h5", "tflite"],
        help="Target format: keras, h5, or tflite"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["none", "int8", "float16", "dynamic_range"],
        help="Quantization mode for TFLite (default: none)"
    )
    parser.add_argument(
        "--representative-data",
        type=str,
        default=None,
        help="Path to .npz file with representative data for int8 quantization"
    )
    
    args = parser.parse_args()
    
    convert_model(
        model_path=args.model_path,
        output_format=args.format,
        output_path=args.output,
        quantization=args.quantization,
        representative_data_path=args.representative_data
    )

