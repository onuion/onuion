"""
Feature Extractor Tests
"""

import pytest
import numpy as np
from onuion.feature_extractor import FeatureExtractor


def test_feature_extractor_basic():
    """Basic feature extraction test."""
    extractor = FeatureExtractor()

    session_data = {
        "current_ip": "192.168.1.100",
        "initial_ip": "192.168.1.50",
        "ip_history": ["192.168.1.50", "192.168.1.100"],
        "current_geo": {"country": "TR", "city": "Istanbul"},
        "initial_geo": {"country": "TR", "city": "Ankara"},
        "current_device": {"fingerprint": "fp1"},
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

    features = extractor.extract(session_data)

    assert isinstance(features, np.ndarray)
    assert features.dtype == np.float32
    assert len(features) == extractor.get_feature_count()
    assert not np.any(np.isnan(features))
    assert not np.any(np.isinf(features))


def test_feature_extractor_empty_data():
    """Empty session data test."""
    extractor = FeatureExtractor()

    session_data = {}

    features = extractor.extract(session_data)

    assert isinstance(features, np.ndarray)
    assert len(features) == extractor.get_feature_count()
    assert not np.any(np.isnan(features))


def test_feature_extractor_feature_names():
    """Feature isimleri testi."""
    extractor = FeatureExtractor()

    feature_names = extractor._get_feature_names()

    assert len(feature_names) == extractor.get_feature_count()
    assert all(isinstance(name, str) for name in feature_names)
