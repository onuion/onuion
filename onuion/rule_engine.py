"""
Rule Engine Module

Performs fast risk detection using deterministic rules.
Runs before the ML model and identifies suspicious situations.
"""

from typing import Dict, Any, List, Set
import time
import numpy as np


class RuleEngine:
    """
    Rule-based risk detection engine.

    Runs very fast (< 0.1ms) and produces deterministic results.
    Performs fast filtering by running before the ML model.
    """

    def __init__(self):
        """Initializes the rule engine."""
        self.risk_types = {
            "ip_mismatch",
            "session_hijacking",
            "bot_behavior",
            "geo_anomaly",
            "device_fingerprint_mismatch",
            "rapid_ip_change",
            "suspicious_request_pattern",
        }

    def evaluate(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates session data using rule-based rules.

        Args:
            session_data: Session data to analyze

        Returns:
            {
                "detected_risks": List[str],  # Detected risk types
                "risk_score": float,  # Rule-based risk score (0-100)
                "rule_details": Dict[str, bool]  # Result of each rule
            }
        """
        detected_risks: List[str] = []
        rule_details: Dict[str, bool] = {}
        risk_score = 0.0

        # Rule 1: IP Mismatch
        if self._check_ip_mismatch(session_data):
            detected_risks.append("ip_mismatch")
            risk_score += 25.0
            rule_details["ip_mismatch"] = True
        else:
            rule_details["ip_mismatch"] = False

        # Rule 2: Session Hijacking
        if self._check_session_hijacking(session_data):
            detected_risks.append("session_hijacking")
            risk_score += 30.0
            rule_details["session_hijacking"] = True
        else:
            rule_details["session_hijacking"] = False

        # Rule 3: Bot Behavior
        if self._check_bot_behavior(session_data):
            detected_risks.append("bot_behavior")
            risk_score += 20.0
            rule_details["bot_behavior"] = True
        else:
            rule_details["bot_behavior"] = False

        # Rule 4: Geo Anomaly
        if self._check_geo_anomaly(session_data):
            detected_risks.append("geo_anomaly")
            risk_score += 15.0
            rule_details["geo_anomaly"] = True
        else:
            rule_details["geo_anomaly"] = False

        # Rule 5: Device Fingerprint Mismatch
        if self._check_device_fingerprint_mismatch(session_data):
            detected_risks.append("device_fingerprint_mismatch")
            risk_score += 20.0
            rule_details["device_fingerprint_mismatch"] = True
        else:
            rule_details["device_fingerprint_mismatch"] = False

        # Rule 6: Rapid IP Change
        if self._check_rapid_ip_change(session_data):
            detected_risks.append("rapid_ip_change")
            risk_score += 25.0
            rule_details["rapid_ip_change"] = True
        else:
            rule_details["rapid_ip_change"] = False

        # Rule 7: Suspicious Request Pattern
        if self._check_suspicious_request_pattern(session_data):
            detected_risks.append("suspicious_request_pattern")
            risk_score += 15.0
            rule_details["suspicious_request_pattern"] = True
        else:
            rule_details["suspicious_request_pattern"] = False

        # Limit risk score to 0-100 range
        risk_score = min(risk_score, 100.0)

        return {
            "detected_risks": detected_risks,
            "risk_score": risk_score,
            "rule_details": rule_details,
        }

    def _check_ip_mismatch(self, session_data: Dict[str, Any]) -> bool:
        """
        IP change check.

        Risk exists if initial IP differs from current IP.
        """
        current_ip = session_data.get("current_ip", "")
        initial_ip = session_data.get("initial_ip", "")

        if not current_ip or not initial_ip:
            return False

        return current_ip != initial_ip

    def _check_session_hijacking(self, session_data: Dict[str, Any]) -> bool:
        """
        Session hijacking check.

        IP change + device fingerprint change + same session ID = hijacking risk
        """
        current_ip = session_data.get("current_ip", "")
        initial_ip = session_data.get("initial_ip", "")

        current_device = session_data.get("current_device", {})
        initial_device = session_data.get("initial_device", {})

        current_session_id = session_data.get("current_session_id", "")
        initial_session_id = session_data.get("initial_session_id", "")

        # IP changed
        ip_changed = current_ip != initial_ip

        # Device fingerprint changed
        current_fp = current_device.get("fingerprint", "")
        initial_fp = initial_device.get("fingerprint", "")
        fp_changed = current_fp != initial_fp

        # Session ID same (session continues)
        session_continues = current_session_id == initial_session_id and current_session_id != ""

        # If all three are true, there is session hijacking risk
        return ip_changed and fp_changed and session_continues

    def _check_bot_behavior(self, session_data: Dict[str, Any]) -> bool:
        """
        Bot behavior check.

        Very high request rate or very regular request pattern indicates bot behavior.
        """
        requests = session_data.get("requests", [])
        if not requests or len(requests) < 5:
            return False

        session_duration = session_data.get("session_duration_seconds", 1.0)
        if session_duration <= 0:
            return False

        request_count = len(requests)
        request_rate = request_count / session_duration

        # More than 50 requests per second = bot behavior
        if request_rate > 50:
            return True

        # Are intervals between requests very regular? (bot pattern)
        intervals = []
        for i in range(1, len(requests)):
            interval = requests[i].get("timestamp", 0) - requests[i - 1].get("timestamp", 0)
            if interval > 0:
                intervals.append(interval)

        if len(intervals) >= 5:
            # If standard deviation is very low (very regular), it might be a bot
            if len(intervals) > 1:
                std_dev = np.std(intervals)
                mean_interval = np.mean(intervals)
                # If coefficient of variation < 0.1, it's very regular
                if mean_interval > 0 and std_dev / mean_interval < 0.1:
                    return True

        return False

    def _check_geo_anomaly(self, session_data: Dict[str, Any]) -> bool:
        """
        Geo-location anomaly check.

        Risk exists if there is a large difference between initial and current location (country change).
        """
        current_geo = session_data.get("current_geo", {})
        initial_geo = session_data.get("initial_geo", {})

        current_country = current_geo.get("country", "")
        initial_country = initial_geo.get("country", "")

        if not current_country or not initial_country:
            return False

        # Country change is an anomaly
        if current_country != initial_country:
            # Suspicious if session duration is very short (e.g., country change within 1 hour)
            session_duration = session_data.get("session_duration_seconds", 0.0)
            if session_duration < 3600:  # Less than 1 hour
                return True

        return False

    def _check_device_fingerprint_mismatch(self, session_data: Dict[str, Any]) -> bool:
        """
        Device fingerprint mismatch check.

        Risk exists if device fingerprint has changed.
        """
        current_device = session_data.get("current_device", {})
        initial_device = session_data.get("initial_device", {})

        current_fp = current_device.get("fingerprint", "")
        initial_fp = initial_device.get("fingerprint", "")

        if not current_fp or not initial_fp:
            return False

        return current_fp != initial_fp

    def _check_rapid_ip_change(self, session_data: Dict[str, Any]) -> bool:
        """
        Rapid IP change check.

        Too many IP changes in a short time is a risk indicator.
        """
        ip_history = session_data.get("ip_history", [])
        if not ip_history or len(ip_history) < 3:
            return False

        # Unique IP count
        unique_ips = len(set(ip_history))

        # Session duration
        session_duration = session_data.get("session_duration_seconds", 1.0)

        # More than 3 different IPs within 1 hour = risk
        if session_duration < 3600 and unique_ips >= 3:
            return True

        # More than 2 different IPs within 10 minutes = risk
        if session_duration < 600 and unique_ips >= 2:
            return True

        return False

    def _check_suspicious_request_pattern(self, session_data: Dict[str, Any]) -> bool:
        """
        Suspicious request pattern check.

        Too many POST/PUT/DELETE requests or unusual endpoints are risk indicators.
        """
        requests = session_data.get("requests", [])
        if not requests or len(requests) < 3:
            return False

        request_count = len(requests)

        # Ratio of risky HTTP methods
        risky_methods = ["POST", "PUT", "DELETE", "PATCH"]
        risky_count = sum(1 for r in requests if r.get("method", "") in risky_methods)
        risky_ratio = risky_count / request_count

        # Suspicious if more than 50% of requests use risky methods
        if risky_ratio > 0.5:
            return True

        # Too many requests to different endpoints (reconnaissance pattern)
        unique_endpoints = len(set(r.get("endpoint", "") for r in requests))
        if unique_endpoints > request_count * 0.8:  # More than 80% unique endpoints
            return True

        return False
