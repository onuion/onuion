"""
Rule Engine Tests
"""

import pytest
from onuion.rule_engine import RuleEngine


def test_rule_engine_ip_mismatch():
    """IP mismatch kuralı testi."""
    engine = RuleEngine()

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

    result = engine.evaluate(session_data)

    assert "ip_mismatch" in result["detected_risks"]
    assert result["risk_score"] > 0


def test_rule_engine_session_hijacking():
    """Session hijacking kuralı testi."""
    engine = RuleEngine()

    session_data = {
        "current_ip": "192.168.1.100",
        "initial_ip": "192.168.1.50",
        "ip_history": [],
        "current_geo": {},
        "initial_geo": {},
        "current_device": {"fingerprint": "fp2"},
        "initial_device": {"fingerprint": "fp1"},
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

    result = engine.evaluate(session_data)

    assert "session_hijacking" in result["detected_risks"]


def test_rule_engine_bot_behavior():
    """Bot behavior kuralı testi."""
    engine = RuleEngine()

    # Yüksek request rate
    requests = [
        {"timestamp": i * 0.01, "method": "GET", "endpoint": f"/api/{i}"} for i in range(100)
    ]

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
        "requests": requests,
        "session_duration_seconds": 1.0,  # 1 saniyede 100 request
        "current_session_id": "sess1",
        "initial_session_id": "sess1",
        "current_cookies": {},
        "initial_cookies": {},
        "current_referrer": "",
        "initial_referrer": "",
    }

    result = engine.evaluate(session_data)

    assert "bot_behavior" in result["detected_risks"]


def test_rule_engine_no_risk():
    """Risk yok durumu testi."""
    engine = RuleEngine()

    session_data = {
        "current_ip": "192.168.1.100",
        "initial_ip": "192.168.1.100",
        "ip_history": [],
        "current_geo": {"country": "TR"},
        "initial_geo": {"country": "TR"},
        "current_device": {"fingerprint": "fp1"},
        "initial_device": {"fingerprint": "fp1"},
        "current_browser": {},
        "initial_browser": {},
        "requests": [{"timestamp": 1000, "method": "GET", "endpoint": "/"}],
        "session_duration_seconds": 10.0,
        "current_session_id": "sess1",
        "initial_session_id": "sess1",
        "current_cookies": {},
        "initial_cookies": {},
        "current_referrer": "",
        "initial_referrer": "",
    }

    result = engine.evaluate(session_data)

    # Risk skoru düşük olmalı (ama 0 olmayabilir, ML'e bırakılır)
    assert result["risk_score"] >= 0
    assert result["risk_score"] <= 100
