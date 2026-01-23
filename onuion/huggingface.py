"""
Hugging Face Integration Module

Provides functionality to upload and download models from Hugging Face Hub.
"""

import os
import tempfile
from typing import Optional
from pathlib import Path

try:
    from huggingface_hub import HfApi, Repository, upload_folder
    from huggingface_hub.utils import HfFolder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")

from onuion.model import RiskModel
import tensorflow as tf


class HuggingFaceIntegration:
    """
    Hugging Face Hub integration for onuion models.
    
    Allows uploading and downloading models to/from Hugging Face Hub.
    """
    
    def __init__(self):
        """Initializes Hugging Face integration."""
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required. Install with: pip install huggingface_hub"
            )
        self.api = HfApi()
    
    def upload_model(
        self,
        model: RiskModel,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
        commit_message: str = "Upload onuion model"
    ) -> str:
        """
        Uploads a model to Hugging Face Hub.
        
        Args:
            model: RiskModel instance to upload
            repo_id: Repository ID (e.g., "onuion/onuion-model")
            token: Hugging Face token (optional, uses cached token if not provided)
            private: Whether the repository should be private
            commit_message: Commit message for the upload
            
        Returns:
            Repository URL
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required")
        
        # Create temporary directory for model
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            
            # Save model
            model.save(model_path)
            
            # Create README.md for the model
            readme_content = self._create_model_readme(model)
            readme_path = os.path.join(tmpdir, "README.md")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
            
            # Upload to Hugging Face Hub
            repo_url = self.api.create_repo(
                repo_id=repo_id,
                token=token,
                private=private,
                exist_ok=True
            )
            
            # Upload files
            self.api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                token=token,
                commit_message=commit_message
            )
        
        return repo_url
    
    def download_model(
        self,
        repo_id: str,
        local_dir: Optional[str] = None,
        token: Optional[str] = None
    ) -> RiskModel:
        """
        Downloads a model from Hugging Face Hub.
        
        Args:
            repo_id: Repository ID (e.g., "onuion/onuion-model")
            local_dir: Local directory to save the model (optional)
            token: Hugging Face token (optional)
            
        Returns:
            RiskModel instance
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required")
        
        if local_dir is None:
            local_dir = os.path.join(tempfile.gettempdir(), f"onuion_{repo_id.replace('/', '_')}")
        
        # Download model files
        self.api.hf_hub_download(
            repo_id=repo_id,
            filename="model/saved_model.pb",
            local_dir=local_dir,
            token=token
        )
        
        # Load model
        model = RiskModel()
        model_path = os.path.join(local_dir, "model")
        model.load(model_path)
        
        return model
    
    def _create_model_readme(self, model: RiskModel) -> str:
        """Creates a README.md for the Hugging Face model repository."""
        return f"""---
library_name: tensorflow
tags:
- onuion
- security
- risk-analysis
- session-analysis
- machine-learning
license: mit
---

# onuion Risk Analysis Model

This is a pre-trained onuion security risk analysis model.

## Model Details

- **Architecture**: Feed Forward Neural Network
- **Input Features**: 25
- **Parameters**: ~{model.get_parameter_count():,}
- **Output**: Risk probability (0.0 - 1.0)

## Usage

```python
from onuion.huggingface import HuggingFaceIntegration
from onuion import analyze_risk

# Download model
hf = HuggingFaceIntegration()
model = hf.download_model("onuion/onuion-model")

# Use with inference
result = analyze_risk(session_data, model_path="path/to/model")
```

## Model Architecture

- Input: 25 features
- Dense Layer 1: 64 units, ReLU
- Dense Layer 2: 32 units, ReLU
- Dense Layer 3: 16 units, ReLU
- Output: 1 unit, Sigmoid

## Performance

- Inference Time: < 1ms (CPU)
- Throughput: ~1,000-2,000 requests/second

## Citation

If you use this model, please cite:

```bibtex
@software{{onuion,
  title={{onuion: Real-time Security Risk Analysis Model}},
  author={{onuion contributors}},
  year={{2024}},
  url={{https://github.com/onuion/onuion}}
}}
```

## License

MIT License
"""


def upload_to_hub(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False
) -> str:
    """
    Convenience function to upload a saved model to Hugging Face Hub.
    
    Args:
        model_path: Path to saved model directory
        repo_id: Repository ID (e.g., "onuion/onuion-model")
        token: Hugging Face token (optional)
        private: Whether the repository should be private
        
    Returns:
        Repository URL
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    # Load model
    model = RiskModel()
    model.load(model_path)
    
    # Upload
    hf = HuggingFaceIntegration()
    return hf.upload_model(model, repo_id, token, private)


def download_from_hub(
    repo_id: str,
    local_dir: Optional[str] = None,
    token: Optional[str] = None
) -> RiskModel:
    """
    Convenience function to download a model from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID (e.g., "onuion/onuion-model")
        local_dir: Local directory to save the model (optional)
        token: Hugging Face token (optional)
        
    Returns:
        RiskModel instance
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        )
    
    hf = HuggingFaceIntegration()
    return hf.download_model(repo_id, local_dir, token)

