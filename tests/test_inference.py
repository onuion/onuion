"""
Inference Pipeline Tests
"""

import pytest
from onuion.inference import InferencePipeline, analyze_risk


def test_inference_pipeline_basic():
    """Basic inference test."""
    pipeline = InferencePipeline()

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
        "current_session_id": "sess1",
        "initial_session_id": "sess1",
        "current_cookies": {},
        "initial_cookies": {},
        "current_referrer": "",
        "initial_referrer": "",
    }

    result = pipeline.analyze(session_data)

    assert result.riskScore >= 0
    assert result.riskScore <= 100
    assert isinstance(result.risk, list)
    assert result.inference_time_ms >= 0


def test_analyze_risk_function():
    """Convenience function test."""
    session_data = {
        "current_ip": "192.168.1.100",
        "initial_ip": "192.168.1.100",
        "ip_history": [],
        "current_geo": {},
        "initial_geo": {},
        "current_device": {},
        "initial_device": {},
        "current_browser": {},
        "initial_browser": {},
        "requests": [],
        "session_duration_seconds": 10.0,
        "current_session_id": "sess1",
        "initial_session_id": "sess1",
        "current_cookies": {},
        "initial_cookies": {},
        "current_referrer": "",
        "initial_referrer": "",
    }

    result = analyze_risk(session_data)

    assert result.riskScore >= 0
    assert result.riskScore <= 100
    assert result.inference_time_ms >= 0
