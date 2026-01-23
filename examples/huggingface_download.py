"""
Hugging Face Download Example

Example script to download a model from Hugging Face Hub.
"""

import os
from onuion.huggingface import download_from_hub, HuggingFaceIntegration
from onuion import analyze_risk

def download_and_use():
    """Download a model from Hugging Face and use it for inference."""
    
    # Repository ID (e.g., "onuion/onuion-model")
    repo_id = "onuion/onuion-model"
    
    # Optional: specify local directory to save model
    local_dir = "models/downloaded_model"
    
    # Optional: token if repository is private
    token = os.getenv("HF_TOKEN")
    
    print(f"Downloading model from {repo_id}...")
    
    # Option 1: Use convenience function
    model = download_from_hub(
        repo_id=repo_id,
        local_dir=local_dir,
        token=token
    )
    
    print(f"Model downloaded to {local_dir}")
    
    # Option 2: Use HuggingFaceIntegration class
    # hf = HuggingFaceIntegration()
    # model = hf.download_model(repo_id, local_dir, token)
    
    # Use the model for inference
    session_data = {
        "current_ip": "192.168.1.100",
        "initial_ip": "192.168.1.50",
        "ip_history": [],
        "current_geo": {},
        "initial_geo": {},
        "current_device": {},
        "initial_device": {},
        "current_browser": {},
        "initial_browser": {},
        "requests": [],
        "session_duration_seconds": 10.0,
        "current_session_id": "sess_123",
        "initial_session_id": "sess_123",
        "current_cookies": {},
        "initial_cookies": {},
        "current_referrer": "",
        "initial_referrer": ""
    }
    
    # Save model locally and use with analyze_risk
    model.save(local_dir)
    result = analyze_risk(session_data, model_path=local_dir)
    
    print(f"Risk Score: {result.riskScore}/100")
    print(f"Risks: {result.risk}")


if __name__ == "__main__":
    # Before running, make sure you're logged in (if needed):
    # huggingface-cli login
    # Or set HF_TOKEN environment variable for private repos
    
    download_and_use()

