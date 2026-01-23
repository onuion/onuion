# Risk Types

onuion detects various security risks by analyzing session data. Detected risks are returned as strings in the `risk` array.

## Rule-Based Risk Types

### 1. `ip_mismatch`

**Description**: The IP address at session start differs from the current IP address.

**Detection Rule**: `current_ip != initial_ip`

**Risk Level**: Medium (25 points)

**Example Scenario**:
```json
{
  "initial_ip": "192.168.1.50",
  "current_ip": "192.168.1.100"
}
```

### 2. `session_hijacking`

**Description**: IP change + device fingerprint change + same session ID (session continues).

**Detection Rule**: 
- IP changed
- Device fingerprint changed
- Session ID same

**Risk Level**: High (30 points)

**Example Scenario**:
```json
{
  "initial_ip": "192.168.1.50",
  "current_ip": "192.168.1.100",
  "initial_device": {"fingerprint": "fp1"},
  "current_device": {"fingerprint": "fp2"},
  "current_session_id": "sess_123",
  "initial_session_id": "sess_123"
}
```

### 3. `bot_behavior`

**Description**: Bot-like behavior patterns detected.

**Detection Rule**:
- Request rate > 50 req/s
- OR intervals between requests are very regular (CV < 0.1)

**Risk Level**: Medium (20 points)

**Example Scenario**:
- 100+ requests in 1 second
- Exactly 0.1 seconds between each request

### 4. `geo_anomaly`

**Description**: Geo-location anomaly (country change in short time).

**Detection Rule**:
- Country changed
- AND session duration < 1 hour

**Risk Level**: Medium (15 points)

**Example Scenario**:
```json
{
  "initial_geo": {"country": "TR"},
  "current_geo": {"country": "US"},
  "session_duration_seconds": 1800  // 30 minutes
}
```

### 5. `device_fingerprint_mismatch`

**Description**: Device fingerprint changed.

**Detection Rule**: `current_device.fingerprint != initial_device.fingerprint`

**Risk Level**: Medium (20 points)

**Example Scenario**:
```json
{
  "initial_device": {"fingerprint": "fp1"},
  "current_device": {"fingerprint": "fp2"}
}
```

### 6. `rapid_ip_change`

**Description**: Too many IP changes in a short time.

**Detection Rule**:
- 3+ different IPs within 1 hour
- OR 2+ different IPs within 10 minutes

**Risk Level**: High (25 points)

**Example Scenario**:
```json
{
  "ip_history": ["192.168.1.50", "192.168.1.75", "192.168.1.100", "192.168.1.125"],
  "session_duration_seconds": 1800  // 30 minutes
}
```

### 7. `suspicious_request_pattern`

**Description**: Suspicious request patterns.

**Detection Rule**:
- More than 50% of requests are POST/PUT/DELETE/PATCH
- OR too many unique endpoints (more than 80%)

**Risk Level**: Medium (15 points)

**Example Scenario**:
- 60 out of 100 requests are POST/DELETE
- 50 requests, 45 different endpoints

## ML-Based Risk Types

### 8. `ml_high_risk`

**Description**: ML model risk probability > 0.7.

**Detection Rule**: `ml_score > 0.7`

**Risk Level**: High

### 9. `ml_critical_risk`

**Description**: ML model risk probability > 0.9.

**Detection Rule**: `ml_score > 0.9`

**Risk Level**: Critical

## Risk Score Calculation

Final risk score is a weighted combination of rule-based and ML-based scores:

```
riskScore = (rule_weight × rule_score) + (ml_weight × ml_score)
```

Default weights:
- `rule_weight`: 0.4 (40%)
- `ml_weight`: 0.6 (60%)

## Risk Array

All detected risks are collected in the `risk` array:

```python
result.risk = ["ip_mismatch", "geo_anomaly", "ml_high_risk"]
```

## Risk Levels Summary

| Risk Type | Level | Rule Score |
|-----------|-------|------------|
| `session_hijacking` | High | 30 |
| `rapid_ip_change` | High | 25 |
| `ip_mismatch` | Medium | 25 |
| `bot_behavior` | Medium | 20 |
| `device_fingerprint_mismatch` | Medium | 20 |
| `geo_anomaly` | Medium | 15 |
| `suspicious_request_pattern` | Medium | 15 |
| `ml_high_risk` | High | - |
| `ml_critical_risk` | Critical | - |
