"""
Feature Extractor Module

Converts session data into numeric and categorical features.
Generates metrics such as IP changes, geo differences, fingerprints, request rates, etc.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib


class FeatureExtractor:
    """
    Extracts features from session data for risk analysis.
    
    Generated features:
    - IP change metrics
    - Geo-location differences
    - Device fingerprint changes
    - Request rates and patterns
    - Session timing information
    """
    
    def __init__(self):
        """Initializes the feature extractor."""
        self.feature_names = self._get_feature_names()
    
    def extract(self, session_data: Dict[str, Any]) -> np.ndarray:
        """
        Extracts feature vector from session data.
        
        Args:
            session_data: Session data to analyze
            
        Returns:
            Normalized feature vector (1D numpy array)
        """
        features = []
        
        # IP and Geo features
        features.extend(self._extract_ip_features(session_data))
        features.extend(self._extract_geo_features(session_data))
        
        # Device and Browser features
        features.extend(self._extract_device_features(session_data))
        features.extend(self._extract_browser_features(session_data))
        
        # Request pattern features
        features.extend(self._extract_request_features(session_data))
        
        # Timing features
        features.extend(self._extract_timing_features(session_data))
        
        # Session metadata features
        features.extend(self._extract_session_features(session_data))
        
        feature_vector = np.array(features, dtype=np.float32)
        
        # Clean NaN and inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return feature_vector
    
    def _extract_ip_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts IP change and pattern features."""
        features = []
        
        # IP history
        ip_history = session_data.get("ip_history", [])
        current_ip = session_data.get("current_ip", "")
        initial_ip = session_data.get("initial_ip", "")
        
        # IP change count (normalized: 0-1 range, max 10 changes = 1.0)
        ip_changes = len(set(ip_history)) if ip_history else 0
        features.append(min(ip_changes / 10.0, 1.0))
        
        # Did IP change? (boolean -> float)
        ip_changed = 1.0 if current_ip != initial_ip else 0.0
        features.append(ip_changed)
        
        # IP history length (normalized)
        ip_history_len = len(ip_history) if ip_history else 0
        features.append(min(ip_history_len / 20.0, 1.0))
        
        # IP subnet change (are first 3 octets different?)
        if current_ip and initial_ip:
            current_subnet = ".".join(current_ip.split(".")[:3])
            initial_subnet = ".".join(initial_ip.split(".")[:3])
            subnet_changed = 1.0 if current_subnet != initial_subnet else 0.0
        else:
            subnet_changed = 0.0
        features.append(subnet_changed)
        
        return features
    
    def _extract_geo_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts geo-location features."""
        features = []
        
        current_geo = session_data.get("current_geo", {})
        initial_geo = session_data.get("initial_geo", {})
        
        # Did country change?
        current_country = current_geo.get("country", "")
        initial_country = initial_geo.get("country", "")
        country_changed = 1.0 if current_country != initial_country else 0.0
        features.append(country_changed)
        
        # Did city change?
        current_city = current_geo.get("city", "")
        initial_city = initial_geo.get("city", "")
        city_changed = 1.0 if current_city != initial_city else 0.0
        features.append(city_changed)
        
        # Calculate distance (simple approach: large distance if countries differ)
        if current_country and initial_country:
            distance_score = 1.0 if current_country != initial_country else 0.0
        else:
            distance_score = 0.0
        features.append(distance_score)
        
        # Timezone change
        current_tz = current_geo.get("timezone", "")
        initial_tz = initial_geo.get("timezone", "")
        tz_changed = 1.0 if current_tz != initial_tz else 0.0
        features.append(tz_changed)
        
        return features
    
    def _extract_device_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts device fingerprint features."""
        features = []
        
        current_device = session_data.get("current_device", {})
        initial_device = session_data.get("initial_device", {})
        
        # Did User-Agent change?
        current_ua = current_device.get("user_agent", "")
        initial_ua = initial_device.get("user_agent", "")
        ua_changed = 1.0 if current_ua != initial_ua else 0.0
        features.append(ua_changed)
        
        # Did screen resolution change?
        current_res = current_device.get("screen_resolution", "")
        initial_res = initial_device.get("screen_resolution", "")
        res_changed = 1.0 if current_res != initial_res else 0.0
        features.append(res_changed)
        
        # Did platform change?
        current_platform = current_device.get("platform", "")
        initial_platform = initial_device.get("platform", "")
        platform_changed = 1.0 if current_platform != initial_platform else 0.0
        features.append(platform_changed)
        
        # Device fingerprint hash change
        current_fp = current_device.get("fingerprint", "")
        initial_fp = initial_device.get("fingerprint", "")
        fp_changed = 1.0 if current_fp != initial_fp else 0.0
        features.append(fp_changed)
        
        return features
    
    def _extract_browser_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts browser features."""
        features = []
        
        current_browser = session_data.get("current_browser", {})
        initial_browser = session_data.get("initial_browser", {})
        
        # Did browser name change?
        current_name = current_browser.get("name", "")
        initial_name = initial_browser.get("name", "")
        browser_changed = 1.0 if current_name != initial_name else 0.0
        features.append(browser_changed)
        
        # Did browser version change?
        current_version = current_browser.get("version", "")
        initial_version = initial_browser.get("version", "")
        version_changed = 1.0 if current_version != initial_version else 0.0
        features.append(version_changed)
        
        # Did language change?
        current_lang = current_browser.get("language", "")
        initial_lang = initial_browser.get("language", "")
        lang_changed = 1.0 if current_lang != initial_lang else 0.0
        features.append(lang_changed)
        
        return features
    
    def _extract_request_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts request pattern features."""
        features = []
        
        requests = session_data.get("requests", [])
        request_count = len(requests) if requests else 0
        
        # Request count (normalized: 0-1 range, max 1000 requests = 1.0)
        features.append(min(request_count / 1000.0, 1.0))
        
        # Request rate (requests per second)
        session_duration = session_data.get("session_duration_seconds", 1.0)
        if session_duration > 0:
            request_rate = request_count / session_duration
            # Normalize: 0-100 req/s = 1.0
            features.append(min(request_rate / 100.0, 1.0))
        else:
            features.append(0.0)
        
        # Unique endpoint count
        if requests:
            unique_endpoints = len(set(r.get("endpoint", "") for r in requests))
            features.append(min(unique_endpoints / 50.0, 1.0))
        else:
            features.append(0.0)
        
        # POST/PUT/DELETE ratio (risky operations)
        if requests:
            risky_methods = ["POST", "PUT", "DELETE", "PATCH"]
            risky_count = sum(1 for r in requests if r.get("method", "") in risky_methods)
            risky_ratio = risky_count / request_count
            features.append(risky_ratio)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_timing_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts timing pattern features."""
        features = []
        
        # Session duration (normalized: 0-1 range, max 3600 seconds = 1.0)
        session_duration = session_data.get("session_duration_seconds", 0.0)
        features.append(min(session_duration / 3600.0, 1.0))
        
        # Time between first and last request
        requests = session_data.get("requests", [])
        if requests and len(requests) > 1:
            first_time = requests[0].get("timestamp", 0)
            last_time = requests[-1].get("timestamp", 0)
            time_span = last_time - first_time
            features.append(min(time_span / 3600.0, 1.0))
        else:
            features.append(0.0)
        
        # Average time between requests (for anomaly detection)
        if requests and len(requests) > 1:
            intervals = []
            for i in range(1, len(requests)):
                interval = requests[i].get("timestamp", 0) - requests[i-1].get("timestamp", 0)
                if interval > 0:
                    intervals.append(interval)
            
            if intervals:
                avg_interval = np.mean(intervals)
                # Very short intervals may indicate bot behavior
                # Intervals shorter than 0.1 seconds are suspicious
                suspicious_interval_ratio = sum(1 for iv in intervals if iv < 0.1) / len(intervals)
                features.append(suspicious_interval_ratio)
            else:
                features.append(0.0)
        else:
            features.append(0.0)
        
        return features
    
    def _extract_session_features(self, session_data: Dict[str, Any]) -> List[float]:
        """Extracts session metadata features."""
        features = []
        
        # Session ID change
        current_session_id = session_data.get("current_session_id", "")
        initial_session_id = session_data.get("initial_session_id", "")
        session_id_changed = 1.0 if current_session_id != initial_session_id else 0.0
        features.append(session_id_changed)
        
        # Cookie change
        current_cookies = session_data.get("current_cookies", {})
        initial_cookies = session_data.get("initial_cookies", {})
        cookie_keys_changed = len(set(current_cookies.keys()) - set(initial_cookies.keys()))
        features.append(min(cookie_keys_changed / 10.0, 1.0))
        
        # Referrer change
        current_referrer = session_data.get("current_referrer", "")
        initial_referrer = session_data.get("initial_referrer", "")
        referrer_changed = 1.0 if current_referrer != initial_referrer else 0.0
        features.append(referrer_changed)
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Returns all feature names (for debugging and interpretability)."""
        return [
            # IP features (4)
            "ip_changes", "ip_changed", "ip_history_len", "subnet_changed",
            # Geo features (4)
            "country_changed", "city_changed", "distance_score", "timezone_changed",
            # Device features (4)
            "ua_changed", "screen_res_changed", "platform_changed", "fingerprint_changed",
            # Browser features (3)
            "browser_changed", "browser_version_changed", "language_changed",
            # Request features (4)
            "request_count", "request_rate", "unique_endpoints", "risky_method_ratio",
            # Timing features (3)
            "session_duration", "request_time_span", "suspicious_interval_ratio",
            # Session features (3)
            "session_id_changed", "cookie_keys_changed", "referrer_changed",
        ]
    
    def get_feature_count(self) -> int:
        """Returns total number of features."""
        return len(self.feature_names)

