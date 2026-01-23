# Hugging Face Integration

onuion supports uploading and downloading models to/from Hugging Face Hub, making it easy to share and deploy models.

## Installation

Install the Hugging Face Hub library:

```bash
pip install huggingface_hub
```

## Authentication

Before uploading models, you need to authenticate with Hugging Face:

```bash
huggingface-cli login
```

Or set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Uploading Models

### Upload a Saved Model

If you have a trained model saved locally:

```python
from onuion.huggingface import upload_to_hub

upload_to_hub(
    model_path="models/onuion_model",
    repo_id="onuion/onuion-model",
    token=None,  # Uses cached token if not provided
    private=False
)
```

### Upload a RiskModel Instance

Upload directly from a RiskModel instance:

```python
from onuion.huggingface import HuggingFaceIntegration
from onuion.model import RiskModel

# Load or create your model
model = RiskModel(input_dim=25)
# ... train your model ...

# Upload
hf = HuggingFaceIntegration()
repo_url = hf.upload_model(
    model=model,
    repo_id="onuion/onuion-model",
    token=None,
    private=False,
    commit_message="Upload trained onuion model"
)
```

## Downloading Models

Download a model from Hugging Face Hub:

```python
from onuion.huggingface import download_from_hub
from onuion import analyze_risk

# Download model
model = download_from_hub(
    repo_id="onuion/onuion-model",
    local_dir="models/downloaded_model",
    token=None  # Required only for private repos
)

# Save locally and use
model.save("models/downloaded_model")
result = analyze_risk(session_data, model_path="models/downloaded_model")
```

## Using Downloaded Models

Once downloaded, you can use the model with the standard inference API:

```python
from onuion import analyze_risk

# Use downloaded model
result = analyze_risk(
    session_data,
    model_path="models/downloaded_model"
)

print(f"Risk Score: {result.riskScore}/100")
print(f"Risks: {result.risk}")
```

## Model Repository Structure

When uploaded to Hugging Face Hub, the model repository contains:

```
repo/
├── model/
│   ├── saved_model.pb
│   ├── variables/
│   └── ...
└── README.md
```

The README.md is automatically generated with model information.

## Examples

See the example scripts:

- `examples/huggingface_upload.py` - Upload examples
- `examples/huggingface_download.py` - Download and usage examples

## Best Practices

1. **Version Control**: Use tags or commit messages to version your models
2. **Documentation**: Update the README.md in your repository with model details
3. **Private Repos**: Use private repositories for sensitive models
4. **Model Cards**: Add detailed model cards describing training data and performance

## Troubleshooting

**Import Error**: If you get an import error, make sure `huggingface_hub` is installed:

```bash
pip install huggingface_hub
```

**Authentication Error**: Make sure you're logged in:

```bash
huggingface-cli login
```

**Permission Error**: Check that you have write access to the repository.

