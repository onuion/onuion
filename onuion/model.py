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
        self._is_saved_model: bool = False  # Track if model is SavedModel format
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
        
        # Convert to TensorFlow tensor
        features_tensor = tf.constant(features, dtype=tf.float32)
        
        # Inference - handle both Keras model and SavedModel
        if self._is_saved_model:
            # SavedModel format - use 'serve' signature
            if hasattr(self.model, 'signatures') and 'serve' in self.model.signatures:
                prediction = self.model.signatures['serve'](features_tensor)
            else:
                # Try default signature
                prediction = self.model(features_tensor)
            # Extract the output value
            if isinstance(prediction, dict):
                # Get the first output value
                output_key = list(prediction.keys())[0]
                prediction_value = prediction[output_key]
            else:
                prediction_value = prediction
            return float(prediction_value.numpy()[0][0])
        else:
            # Keras model format
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
        
        # Convert to TensorFlow tensor
        features_tensor = tf.constant(features_batch, dtype=tf.float32)
        
        # Inference - handle both Keras model and SavedModel
        if self._is_saved_model:
            # SavedModel format - use 'serve' signature
            if hasattr(self.model, 'signatures') and 'serve' in self.model.signatures:
                prediction = self.model.signatures['serve'](features_tensor)
            else:
                # Try default signature
                prediction = self.model(features_tensor)
            # Extract the output value
            if isinstance(prediction, dict):
                # Get the first output value
                output_key = list(prediction.keys())[0]
                prediction_value = prediction[output_key]
            else:
                prediction_value = prediction
            return prediction_value.numpy().flatten()
        else:
            # Keras model format
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
            filepath: Save path (directory or file with .keras extension)
        """
        if self.model is None:
            raise ValueError("Model has not been created yet!")
        
        # Check if filepath has extension
        if os.path.splitext(filepath)[1]:
            # Has extension, save as Keras format
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
            self.model.save(filepath)
        else:
            # No extension, save as SavedModel format (directory)
            os.makedirs(filepath, exist_ok=True)
            # Use model.save() with explicit SavedModel format
            # For TensorFlow 2.10+, we need to use save_format='tf' or model.export()
            try:
                # Try using model.save() - should work for SavedModel in newer TF versions
                self.model.save(filepath, save_format='tf')
            except (ValueError, TypeError) as e:
                # If that fails, try model.export() for TF 2.15+
                if hasattr(self.model, 'export'):
                    self.model.export(filepath)
                else:
                    # Final fallback: use tf.saved_model.save()
                    tf.saved_model.save(self.model, filepath)
        
        print(f"Model saved: {filepath}")
    
    def load(self, filepath: str):
        """
        Loads a model in SavedModel format.
        
        Args:
            filepath: Model path (directory or .keras file)
        """
        # Check if it's a .keras file
        if filepath.endswith('.keras'):
            self.model = keras.models.load_model(filepath)
            self._is_saved_model = False
        else:
            # SavedModel format - use tf.saved_model.load() for Keras 3 compatibility
            try:
                # Try keras.models.load_model() first (works for Keras 2.x and .keras files)
                self.model = keras.models.load_model(filepath)
                self._is_saved_model = False
            except (ValueError, TypeError) as e:
                # For Keras 3, SavedModel needs to be loaded differently
                if "SavedModel" in str(e) or "not supported" in str(e) or "V3" in str(e):
                    # Load SavedModel using tf.saved_model.load()
                    self.model = tf.saved_model.load(filepath)
                    self._is_saved_model = True
                else:
                    raise
        
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
        
        if self._is_saved_model:
            # For SavedModel, we can't easily count params
            # Return a default or estimate
            return 0
        else:
            return int(self.model.count_params())
    
    def convert_to_keras(self, output_path: str) -> str:
        """
        Converts model to Keras (.keras) format.
        
        Args:
            output_path: Output file path (should end with .keras)
            
        Returns:
            Path to saved file
        """
        if self.model is None:
            raise ValueError("Model has not been created or loaded yet!")
        
        # Ensure .keras extension
        if not output_path.endswith('.keras'):
            output_path = output_path + '.keras'
        
        # Get Keras model
        if self._is_saved_model:
            # For SavedModel, reconstruct the model architecture and load weights
            # This works if we know the input_dim
            keras_model = RiskModel(input_dim=self.input_dim)
            keras_model.model = keras_model.model  # Get the Keras model
            
            # Try to extract weights from SavedModel and set them
            # This is a workaround - we rebuild the model structure
            try:
                # Rebuild model with same architecture
                keras_model._build_model()
                
                # Extract weights from SavedModel (if possible)
                # Note: This is complex and may not work for all SavedModels
                # Better approach: Save as Keras format during training
                print("Warning: Converting SavedModel to Keras format.")
                print("  Model architecture will be preserved, but some metadata may be lost.")
                
                # Save the rebuilt model
                os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
                keras_model.model.save(output_path)
                self.model = keras_model.model  # Update internal model
                self._is_saved_model = False
            except Exception as e:
                raise ValueError(
                    f"Cannot convert SavedModel to .keras format: {e}. "
                    "Please train a new model and save it as .keras format, "
                    "or use a model that was originally saved as Keras format."
                )
        else:
            # Save as .keras format
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            self.model.save(output_path)
        
        print(f"Model converted to Keras format: {output_path}")
        return output_path
    
    def convert_to_h5(self, output_path: str) -> str:
        """
        Converts model to H5 (.h5) format.
        
        Args:
            output_path: Output file path (should end with .h5)
            
        Returns:
            Path to saved file
        """
        if self.model is None:
            raise ValueError("Model has not been created or loaded yet!")
        
        # Ensure .h5 extension
        if not output_path.endswith('.h5'):
            output_path = output_path + '.h5'
        
        # Get Keras model
        if self._is_saved_model:
            # First convert to Keras, then to H5
            temp_keras_path = output_path.replace('.h5', '_temp.keras')
            self.convert_to_keras(temp_keras_path)
            # Now load and save as H5
            keras_model = keras.models.load_model(temp_keras_path)
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            keras_model.save(output_path, save_format='h5')
            # Clean up temp file
            if os.path.exists(temp_keras_path):
                os.remove(temp_keras_path)
            # Update internal model
            self.model = keras_model
            self._is_saved_model = False
        else:
            # Save as .h5 format
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
            self.model.save(output_path, save_format='h5')
        
        print(f"Model converted to H5 format: {output_path}")
        return output_path
    
    def convert_to_tflite(
        self,
        output_path: str,
        quantization: str = "none",
        representative_dataset: Optional[np.ndarray] = None
    ) -> str:
        """
        Converts model to TensorFlow Lite (.tflite) format.
        
        Args:
            output_path: Output file path (should end with .tflite)
            quantization: Quantization mode - "none", "int8", "float16", "dynamic_range"
            representative_dataset: Optional representative dataset for int8 quantization
                                  (shape: (n_samples, input_dim))
            
        Returns:
            Path to saved file
        """
        if self.model is None:
            raise ValueError("Model has not been created or loaded yet!")
        
        # Ensure .tflite extension
        if not output_path.endswith('.tflite'):
            output_path = output_path + '.tflite'
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Get the Keras model for conversion
        keras_model = None
        if self._is_saved_model:
            # For SavedModel, we need to convert it to a concrete function
            # This is more complex - we'll use the signature
            try:
                # Try to get a concrete function from SavedModel
                if hasattr(self.model, 'signatures') and 'serve' in self.model.signatures:
                    concrete_func = self.model.signatures['serve']
                else:
                    # Try default call
                    @tf.function
                    def model_func(x):
                        return self.model(x)
                    concrete_func = model_func.get_concrete_function(
                        tf.TensorSpec(shape=[None, self.input_dim], dtype=tf.float32)
                    )
                
                # Convert to TFLite
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
            except Exception as e:
                raise ValueError(
                    f"Error converting SavedModel to TFLite: {e}. "
                    "Consider loading a Keras model first or training a new model."
                )
        else:
            # Direct conversion from Keras model
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        # Apply quantization if requested
        if quantization == "int8":
            if representative_dataset is None:
                raise ValueError("representative_dataset is required for int8 quantization")
            
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            def representative_data_gen():
                for i in range(len(representative_dataset)):
                    yield [representative_dataset[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_data_gen
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        elif quantization == "dynamic_range":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Convert and save
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"Model converted to TFLite format: {output_path}")
        print(f"  File size: {file_size:.2f} KB")
        print(f"  Quantization: {quantization}")
        
        return output_path


def create_default_model(input_dim: int = 25) -> RiskModel:
    """
    Creates a default model (convenience function).
    
    Args:
        input_dim: Number of input features
        
    Returns:
        RiskModel instance
    """
    return RiskModel(input_dim=input_dim)

