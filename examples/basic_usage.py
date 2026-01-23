"""
Basic Usage Example

Basic risk analysis example with onuion.
"""

import json
from onuion import analyze_risk

# Example session data
session_data = {
    "current_ip": "192.168.1.100",
    "initial_ip": "192.168.1.50",
    "ip_history": ["192.168.1.50", "192.168.1.100"],
    "current_geo": {
        "country": "TR",
        "city": "Istanbul",
        "timezone": "Europe/Istanbul"
    },
    "initial_geo": {
        "country": "TR",
        "city": "Ankara",
        "timezone": "Europe/Istanbul"
    },
    "current_device": {
        "user_agent": "Mozilla/5.0",
        "screen_resolution": "1920x1080",
        "platform": "Win32",
        "fingerprint": "fp123"
    },
    "initial_device": {
        "user_agent": "Mozilla/5.0",
        "screen_resolution": "1920x1080",
        "platform": "Win32",
        "fingerprint": "fp123"
    },
    "current_browser": {
        "name": "Chrome",
        "version": "120.0",
        "language": "tr-TR"
    },
    "initial_browser": {
        "name": "Chrome",
        "version": "120.0",
        "language": "tr-TR"
    },
    "requests": [
        {"timestamp": 1706000000, "method": "GET", "endpoint": "/api/users"},
        {"timestamp": 1706000005, "method": "POST", "endpoint": "/api/login"},
    ],
    "session_duration_seconds": 5.0,
    "current_session_id": "sess_123",
    "initial_session_id": "sess_123",
    "current_cookies": {},
    "initial_cookies": {},
    "current_referrer": "",
    "initial_referrer": ""
}

# Risk analysis
result = analyze_risk(session_data)

# Print results
print("=" * 50)
print("RISK ANALYSIS RESULTS")
print("=" * 50)
print(f"Risk Score: {result.riskScore}/100")
print(f"Detected Risks: {result.risk}")
print(f"Rule Score: {result.rule_score}/100")
print(f"ML Score: {result.ml_score}/100")
print(f"Confidence: {result.confidence}%")
print(f"Inference Time: {result.inference_time_ms:.3f} ms")
print("=" * 50)

# Print in JSON format
print("\nJSON Format:")
print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

