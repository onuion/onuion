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
risk_session_data = {
    "current_ip": "5.231.26.0",
    "initial_ip": "192.168.1.50",
    "ip_history": ["192.168.1.50", "192.168.1.100"],
    "current_geo": {"country": "US", "city": "New York"},
    "initial_geo": {"country": "TR", "city": "Ankara"},
    "current_device": {"fingerprint": "fp120000"},
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
risk_result = analyze_risk(risk_session_data)

# Results
print(f"Risk Score: {result.riskScore}/100")
print(f"Risks: {result.risk}")
print(f"Inference Time: {result.inference_time_ms:.3f} ms")

# 65% risk score test
print("\nTesting 65% risk score scenario...")
print(f"Risk Score: {risk_result.riskScore}/100")
print(f"Risks: {risk_result.risk}")
print(f"Inference Time: {risk_result.inference_time_ms:.3f} ms")

