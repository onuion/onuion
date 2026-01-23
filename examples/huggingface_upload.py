"""
Hugging Face Upload Example

Example script to upload a trained onuion model to Hugging Face Hub.
"""

import os
from onuion.huggingface import upload_to_hub, HuggingFaceIntegration
from onuion.model import RiskModel

# Option 1: Upload from saved model directory
def upload_saved_model():
    """Upload a previously saved model."""
    model_path = "models/onuion_model"  # Path to your saved model
    
    # Get token from environment or Hugging Face CLI
    token = os.getenv("HF_TOKEN")  # Or use: huggingface-cli login
    
    repo_id = "onuion/onuion"  # Hugging Face repository ID
    
    print(f"Uploading model from {model_path} to {repo_id}...")
    repo_url = upload_to_hub(
        model_path=model_path,
        repo_id=repo_id,
        token=token,
        private=False  # Set to True for private repository
    )
    
    print(f"Model uploaded successfully!")
    print(f"Repository URL: {repo_url}")


# Option 2: Upload directly from RiskModel instance
def upload_model_instance():
    """Upload a RiskModel instance directly."""
    # Load or create your model
    model = RiskModel(input_dim=25)
    # ... train your model ...
    
    # Upload
    hf = HuggingFaceIntegration()
    token = os.getenv("HF_TOKEN")
    repo_id = "onuion/onuion
    
    print(f"Uploading model to {repo_id}...")
    repo_url = hf.upload_model(
        model=model,
        repo_id=repo_id,
        token=token,
        private=False,
        commit_message="Upload trained onuion model"
    )
    
    print(f"Model uploaded successfully!")
    print(f"Repository URL: {repo_url}")


if __name__ == "__main__":
    # Before running, make sure you're logged in:
    # huggingface-cli login
    # Or set HF_TOKEN environment variable
    
    # Choose one:
    # upload_saved_model()
    # upload_model_instance()
    
    print("See the functions above for usage examples.")
    print("Make sure to:")
    print("1. Train your model first")
    print("2. Login to Hugging Face: huggingface-cli login")
    print("3. Update repo_id if needed (default: onuion/onuion)")

