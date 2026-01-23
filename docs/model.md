# Model Architecture

## Overview

onuion uses a small Feed Forward Neural Network (FFN) optimized for tabular data. The model is designed for real-time inference and targets sub-millisecond inference time.

## Architecture Details

### Input Layer

- **Shape**: `(batch_size, 25)`
- **Features**: 25 numeric features from feature extractor
- **Normalization**: Features already come normalized

### Hidden Layers

#### Layer 1: Dense + BatchNorm + Dropout
- **Units**: 64
- **Activation**: ReLU
- **Batch Normalization**: Yes
- **Dropout**: 0.2

#### Layer 2: Dense + BatchNorm + Dropout
- **Units**: 32
- **Activation**: ReLU
- **Batch Normalization**: Yes
- **Dropout**: 0.2

#### Layer 3: Dense
- **Units**: 16
- **Activation**: ReLU

### Output Layer

- **Units**: 1
- **Activation**: Sigmoid
- **Output**: Risk probability (0.0 - 1.0)

## Parameter Count

Total trainable parameters: ~2,000

- Layer 1: 25 × 64 + 64 = 1,664
- Layer 2: 64 × 32 + 32 = 2,080
- Layer 3: 32 × 16 + 16 = 528
- Output: 16 × 1 + 1 = 17
- **Total**: ~4,289 parameters (batch norm and dropout not included)

## Why This Architecture?

1. **Small Model**: Minimal parameters for fast inference
2. **Feed Forward**: Most suitable architecture for tabular data
3. **Batch Normalization**: For training stability and speed
4. **Dropout**: Prevents overfitting
5. **Sigmoid Output**: Ideal for binary risk probability

## Why Not Use Transformer/RNN?

- **Transformer**: Designed for sequence data, unnecessary complexity for tabular data
- **RNN/LSTM**: Not suitable as sequential patterns are not needed
- **CNN**: For image data, not suitable for tabular data

FFN is the most suitable and fastest option for tabular data.

## Training

### Loss Function

- **Binary Crossentropy**: Standard for binary classification

### Optimizer

- **Adam**: Learning rate 0.001
- **Learning Rate Scheduling**: ReduceLROnPlateau

### Callbacks

- **Early Stopping**: Based on validation loss, patience=10
- **ReduceLROnPlateau**: Learning rate reduction, patience=5

## Inference Optimization

1. **TensorFlow Graph Mode**: Graph mode instead of eager mode
2. **Batch Prediction**: Batch inference when possible
3. **Model Quantization**: Optional (future version)
4. **ONNX Export**: For cross-platform (future version)

## Model Saving/Loading

Model is saved in SavedModel format:

```python
model.save("models/onuion_model")
model.load("models/onuion_model")
```

## Performance

- **Inference Time**: < 1ms (CPU, single inference)
- **Throughput**: ~1,000-2,000 requests/second
- **Model Size**: ~50 KB (saved model)
