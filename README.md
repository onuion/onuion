# onuion

**Real-time Security Risk Analysis Model**

Open-source hybrid (rule-based + ML) security risk analysis system for session data analysis with sub-millisecond inference time.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Hub-yellow)](https://huggingface.co/onuion/onuion)

## 🎯 Purpose

`onuion` is a production-ready risk analysis system that detects security risks by analyzing session data. Using a hybrid approach (rule-based + ML), it provides both fast detection with deterministic rules and sophisticated pattern recognition with machine learning.

## ⚡ Features

- **Real-time Inference**: < 1ms inference time target
- **Hybrid System**: Rule-based + TensorFlow ML model
- **Small Model**: ~2,000 parameters, optimized for tabular data
- **Production-Ready**: Modular, tested, documented code
- **Open Source**: Licensed under MIT

## 🏗️ Architecture

```
Session Data
    ↓
[Feature Extractor] → 25 numeric/categorical features
    ↓
[Rule Engine] → Deterministic risk detection (< 0.1ms)
    ↓
[TensorFlow Model] → ML-based risk probability (< 0.5ms)
    ↓
[Risk Aggregator] → Final riskScore + risk array
    ↓
Risk Analysis Result
```

### Components

1. **Feature Extractor**: Converts session data into 25 features
   - IP change metrics
   - Geo-location differences
   - Device fingerprint changes
   - Request rates and patterns
   - Session timing information

2. **Rule Engine**: Fast risk detection with deterministic rules
   - `ip_mismatch`: IP change
   - `session_hijacking`: Session hijacking indicators
   - `bot_behavior`: Bot behavior patterns
   - `geo_anomaly`: Geo-location anomaly
   - `device_fingerprint_mismatch`: Device fingerprint mismatch
   - `rapid_ip_change`: Rapid IP change
   - `suspicious_request_pattern`: Suspicious request patterns

3. **TensorFlow Model**: Feed Forward Neural Network
   - Input: 25 features
   - 3 Dense layers (64 → 32 → 16 units)
   - Output: Risk probability (0.0 - 1.0)
   - Total parameters: ~2,000

4. **Risk Aggregator**: Combines rule and ML scores
   - Weighted combination (Rule 40%, ML 60%)
   - Final riskScore (0-100)
   - Risk array generation

## 📦 Installation

### Requirements

- Python 3.8+
- TensorFlow 2.10+
- NumPy 1.21+

### Installation Steps

```bash
# Clone repository
git clone https://github.com/onuion/onuion.git
cd onuion

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

## 🚀 Usage

### Basic Usage

```python
from onuion import analyze_risk

# Session data
session_data = {
    "current_ip": "192.168.1.100",
    "initial_ip": "192.168.1.50",
    "ip_history": ["192.168.1.50", "192.168.1.100"],
    "current_geo": {"country": "TR", "city": "Istanbul"},
    "initial_geo": {"country": "TR", "city": "Ankara"},
    "current_device": {"fingerprint": "fp123"},
    "initial_device": {"fingerprint": "fp123"},
    "current_browser": {},
    "initial_browser": {},
    "requests": [
        {"timestamp": 1706000000, "method": "GET", "endpoint": "/api/users"}
    ],
    "session_duration_seconds": 10.0,
    "current_session_id": "sess_123",
    "initial_session_id": "sess_123",
    "current_cookies": {},
    "initial_cookies": {},
    "current_referrer": "",
    "initial_referrer": ""
}

# Risk analysis
result = analyze_risk(session_data)

# Results
print(f"Risk Score: {result.riskScore}/100")
print(f"Risks: {result.risk}")
print(f"Inference Time: {result.inference_time_ms:.3f} ms")
```

### Output Format

```python
{
    "riskScore": 45.2,  # Risk score (0-100)
    "risk": ["ip_mismatch", "geo_anomaly"],  # Detected risk types
    "rule_score": 40.0,  # Rule engine score
    "ml_score": 48.0,  # ML model score
    "confidence": 92.0,  # Confidence score
    "inference_time_ms": 0.856  # Inference time (ms)
}
```

### Examples

See the `examples/` directory for more examples:

- `examples/basic_usage.py`: Basic usage example
- `examples/load_from_json.py`: Load from JSON file
- `examples/sample_session.json`: Example session data

## 🧪 Model Training

### Training with Synthetic Data (for testing)

```bash
python -m onuion.train --synthetic --epochs 50 --output-dir models/onuion_model
```

### Training with Real Data

```bash
# Data format: .npz file
# Contents: X_train, y_train, X_val, y_val (numpy arrays)

python -m onuion.train --data-path data/training_data.npz --epochs 100
```

### Hugging Face Integration

onuion supports Hugging Face Hub for model sharing and deployment.

**Model Hub**: [https://huggingface.co/onuion/onuion](https://huggingface.co/onuion/onuion)

**Upload a model to Hugging Face Hub:**

```python
from onuion.huggingface import upload_to_hub

# Upload a saved model
upload_to_hub(
    model_path="models/onuion_model",
    repo_id="onuion/onuion",
    token="your_hf_token",  # Optional, can use huggingface-cli login
    private=False
)
```

**Download a model from Hugging Face Hub:**

```python
from onuion.huggingface import download_from_hub
from onuion import analyze_risk

# Download model
model = download_from_hub(
    repo_id="onuion/onuion",
    local_dir="models/downloaded_model"
)

# Use with inference
model.save("models/downloaded_model")
result = analyze_risk(session_data, model_path="models/downloaded_model")
```

**Install Hugging Face dependencies:**

```bash
pip install huggingface_hub
```

See `examples/huggingface_upload.py` and `examples/huggingface_download.py` for complete examples.

## 📊 Performance

### Inference Time

- **Target**: < 1ms
- **Average**: ~0.5-0.8ms (CPU)
- **P95**: < 1.5ms
- **Throughput**: ~1,000-2,000 requests/second

### Benchmark

```bash
python benchmark/benchmark.py
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=onuion --cov-report=html
```

## 📚 Documentation

See the `docs/` directory for detailed documentation:

- `docs/model.md`: Model architecture details
- `docs/risk_types.md`: Risk type descriptions
- `docs/api.md`: API reference

## 🤝 Contributing

We welcome your contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

- TensorFlow team
- Open source community
- All contributors

## 📧 Contact

You can open an issue or send a pull request for questions.

---

**Note**: This project is prepared for production use, but it is recommended to use a model trained with real data. The default model is only for testing inference purposes.

