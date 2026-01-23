"""
TensorFlow Model Module

Small, fast Feed Forward Neural Network model.
Optimized for tabular data, targeting sub-millisecond inference time.
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional, Tuple
import numpy as np
import os


class RiskModel:
    """
    TensorFlow model for risk analysis.
    
    Architecture:
    - Input: 25 features (from feature_extractor)
    - Dense Layer 1: 64 units, ReLU
    - Dense Layer 2: 32 units, ReLU
    - Dense Layer 3: 16 units, ReLU
    - Output: 1 unit, Sigmoid (risk probability)
    
    Total parameter count: ~2,000 (very small and fast)
    """
    
    def __init__(self, input_dim: int = 25):
        """
        Initializes the model.
        
        Args:
            input_dim: Number of input features (default: 25)
        """
        self.input_dim = input_dim
        self.model: Optional[keras.Model] = None
        self._build_model()
    
    def _build_model(self):
        """Builds the model architecture."""
        inputs = keras.Input(shape=(self.input_dim,), name="features")
        
        # Dense Layer 1
        x = keras.layers.Dense(
            64,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_1"
        )(inputs)
        x = keras.layers.BatchNormalization(name="bn_1")(x)
        x = keras.layers.Dropout(0.2, name="dropout_1")(x)
        
        # Dense Layer 2
        x = keras.layers.Dense(
            32,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_2"
        )(x)
        x = keras.layers.BatchNormalization(name="bn_2")(x)
        x = keras.layers.Dropout(0.2, name="dropout_2")(x)
        
        # Dense Layer 3
        x = keras.layers.Dense(
            16,
            activation="relu",
            kernel_initializer="he_normal",
            name="dense_3"
        )(x)
        
        # Output Layer (risk probability)
        outputs = keras.layers.Dense(
            1,
            activation="sigmoid",
            name="risk_probability"
        )(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs, name="risk_model")
        
        # Compile the model (for training)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", "precision", "recall"]
        )
    
    def predict(self, features: np.ndarray) -> float:
        """
        Predicts risk probability from feature vector.
        
        Args:
            features: Feature vector (shape: (1, input_dim) or (input_dim,))
            
        Returns:
            Risk probability (0.0 - 1.0)
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet!")
        
        # Shape check and fix
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Inference
        prediction = self.model.predict(features, verbose=0)
        
        return float(prediction[0][0])
    
    def predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """
        Performs batch prediction (more efficient).
        
        Args:
            features_batch: Feature matrix (shape: (batch_size, input_dim))
            
        Returns:
            Risk probabilities array (shape: (batch_size,))
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet!")
        
        predictions = self.model.predict(features_batch, verbose=0)
        
        return predictions.flatten()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """
        Trains the model.
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        if self.model is None:
            raise ValueError("Model has not been created yet!")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def save(self, filepath: str):
        """
        Saves the model in SavedModel format.
        
        Args:
            filepath: Save path (directory)
        """
        if self.model is None:
            raise ValueError("Model has not been created yet!")
        
        os.makedirs(filepath, exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved: {filepath}")
    
    def load(self, filepath: str):
        """
        Loads a model in SavedModel format.
        
        Args:
            filepath: Model path (directory)
        """
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded: {filepath}")
    
    def get_model_summary(self) -> str:
        """Returns model summary as string."""
        if self.model is None:
            return "Model has not been created yet!"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
    
    def get_parameter_count(self) -> int:
        """Returns total number of trainable parameters."""
        if self.model is None:
            return 0
        
        return int(self.model.count_params())


def create_default_model(input_dim: int = 25) -> RiskModel:
    """
    Creates a default model (convenience function).
    
    Args:
        input_dim: Number of input features
        
    Returns:
        RiskModel instance
    """
    return RiskModel(input_dim=input_dim)

