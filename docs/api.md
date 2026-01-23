# API Reference

## Main Module: `onuion`

### `analyze_risk(session_data, model_path=None)`

Analyzes session data and produces risk score.

**Parameters**:
- `session_data` (dict): Session data to analyze
- `model_path` (str, optional): Model path (used on first call)

**Returns**: `RiskAnalysisResult` object

**Example**:
```python
from onuion import analyze_risk

result = analyze_risk(session_data)
print(result.riskScore)
```

### `RiskModel`

TensorFlow model class.

**Methods**:
- `predict(features)`: Predicts risk probability
- `predict_batch(features_batch)`: Batch prediction
- `train(X_train, y_train, ...)`: Trains the model
- `save(filepath)`: Saves the model
- `load(filepath)`: Loads the model

## Modules

### `onuion.feature_extractor`

#### `FeatureExtractor`

Extracts features from session data.

**Methods**:
- `extract(session_data)`: Returns feature vector
- `get_feature_count()`: Returns number of features

### `onuion.rule_engine`

#### `RuleEngine`

Rule-based risk detection.

**Methods**:
- `evaluate(session_data)`: Performs rule evaluation

**Returns**:
```python
{
    "detected_risks": List[str],
    "risk_score": float,
    "rule_details": Dict[str, bool]
}
```

### `onuion.model`

#### `RiskModel`

TensorFlow model class.

### `onuion.risk_aggregator`

#### `RiskAggregator`

Combines rule and ML scores.

**Methods**:
- `aggregate(rule_result, ml_score)`: Combines scores

### `onuion.inference`

#### `InferencePipeline`

End-to-end inference pipeline.

**Methods**:
- `analyze(session_data)`: Performs risk analysis
- `analyze_batch(session_data_list)`: Batch inference

#### `RiskAnalysisResult`

Risk analysis result.

**Properties**:
- `riskScore` (float): Final risk score (0-100)
- `risk` (list): Detected risk types
- `rule_score` (float): Rule engine score
- `ml_score` (float): ML model score
- `confidence` (float): Confidence score
- `inference_time_ms` (float): Inference time

**Methods**:
- `to_dict()`: Converts to dictionary format

## Session Data Format

```python
{
    "current_ip": str,
    "initial_ip": str,
    "ip_history": List[str],
    "current_geo": {
        "country": str,
        "city": str,
        "timezone": str
    },
    "initial_geo": {
        "country": str,
        "city": str,
        "timezone": str
    },
    "current_device": {
        "user_agent": str,
        "screen_resolution": str,
        "platform": str,
        "fingerprint": str
    },
    "initial_device": {
        "user_agent": str,
        "screen_resolution": str,
        "platform": str,
        "fingerprint": str
    },
    "current_browser": {
        "name": str,
        "version": str,
        "language": str
    },
    "initial_browser": {
        "name": str,
        "version": str,
        "language": str
    },
    "requests": List[{
        "timestamp": int,
        "method": str,
        "endpoint": str
    }],
    "session_duration_seconds": float,
    "current_session_id": str,
    "initial_session_id": str,
    "current_cookies": dict,
    "initial_cookies": dict,
    "current_referrer": str,
    "initial_referrer": str
}
```

## Example Usage

```python
from onuion import analyze_risk, RiskModel
from onuion.inference import InferencePipeline

# Simple usage
result = analyze_risk(session_data)

# With pipeline
pipeline = InferencePipeline(model_path="models/onuion_model")
result = pipeline.analyze(session_data)

# Batch inference
results = pipeline.analyze_batch([session1, session2, session3])

# Model training
from onuion.model import RiskModel
model = RiskModel(input_dim=25)
model.train(X_train, y_train, X_val, y_val, epochs=50)
model.save("models/onuion_model")
```
