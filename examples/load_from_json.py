"""
Load from JSON File Example

Loads session data from sample_session.json and analyzes it.
"""

import json
import os
from onuion import analyze_risk

# Load JSON file
json_path = os.path.join(os.path.dirname(__file__), "sample_session.json")

with open(json_path, "r", encoding="utf-8") as f:
    session_data = json.load(f)

# Risk analysis
result = analyze_risk(session_data)

# Print results
print("Session Data Analysis")
print("-" * 50)
print(f"IP: {session_data['initial_ip']} -> {session_data['current_ip']}")
print(f"Geo: {session_data['initial_geo']['city']} -> {session_data['current_geo']['city']}")
print(f"Session ID: {session_data['current_session_id']}")
print("-" * 50)
print(f"\nRisk Score: {result.riskScore}/100")
print(f"Risks: {', '.join(result.risk) if result.risk else 'None'}")
print(f"Inference: {result.inference_time_ms:.3f} ms")

