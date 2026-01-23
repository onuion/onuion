"""
Risk Aggregator Module

Combines rule engine results with ML model score.
Produces final riskScore and risk array.
"""

from typing import Dict, Any, List
import numpy as np


class RiskAggregator:
    """
    Combines rule-based and ML-based risk scores.

    Strategy:
    - Rule engine results: Deterministic, fast
    - ML model score: Pattern-based, more sophisticated
    - Weighted combination: Rule 40%, ML 60% (adjustable)
    """

    def __init__(self, rule_weight: float = 0.4, ml_weight: float = 0.6):
        """
        Initializes the risk aggregator.

        Args:
            rule_weight: Weight of rule engine score (0-1)
            ml_weight: Weight of ML model score (0-1)
        """
        if abs(rule_weight + ml_weight - 1.0) > 0.01:
            raise ValueError("rule_weight + ml_weight must equal 1.0!")

        self.rule_weight = rule_weight
        self.ml_weight = ml_weight

    def aggregate(self, rule_result: Dict[str, Any], ml_score: float) -> Dict[str, Any]:
        """
        Combines rule and ML scores.

        Args:
            rule_result: Result from rule engine
                {
                    "detected_risks": List[str],
                    "risk_score": float (0-100),
                    "rule_details": Dict[str, bool]
                }
            ml_score: Risk probability from ML model (0.0-1.0)

        Returns:
            {
                "riskScore": float (0-100),
                "risk": List[str],
                "rule_score": float,
                "ml_score": float,
                "confidence": float
            }
        """
        rule_score = rule_result.get("risk_score", 0.0)
        detected_risks = rule_result.get("detected_risks", [])

        # Convert ML score to 0-100 range
        ml_score_100 = ml_score * 100.0

        # Weighted combination
        final_score = (self.rule_weight * rule_score) + (self.ml_weight * ml_score_100)

        # Limit to 0-100 range
        final_score = max(0.0, min(100.0, final_score))

        # Build risk array
        risk_array = self._build_risk_array(rule_result, ml_score)

        # Calculate confidence (agreement between rule and ML scores)
        confidence = self._calculate_confidence(rule_score, ml_score_100)

        return {
            "riskScore": round(final_score, 2),
            "risk": risk_array,
            "rule_score": round(rule_score, 2),
            "ml_score": round(ml_score_100, 2),
            "confidence": round(confidence, 2),
        }

    def _build_risk_array(self, rule_result: Dict[str, Any], ml_score: float) -> List[str]:
        """
        Builds the final risk array.

        Risks from rule engine + additional risks based on ML threshold.
        """
        risk_array = []

        # Risks from rule engine
        detected_risks = rule_result.get("detected_risks", [])
        risk_array.extend(detected_risks)

        # Add general risk if ML score is high (threshold: 0.7)
        if ml_score > 0.7 and "ml_high_risk" not in risk_array:
            risk_array.append("ml_high_risk")

        # Add critical risk if ML score is very high (threshold: 0.9)
        if ml_score > 0.9 and "ml_critical_risk" not in risk_array:
            risk_array.append("ml_critical_risk")

        # Remove duplicates and sort
        risk_array = sorted(list(set(risk_array)))

        return risk_array

    def _calculate_confidence(self, rule_score: float, ml_score: float) -> float:
        """
        Calculates confidence score.

        Confidence is high if rule and ML scores are close.
        Confidence is low if they differ.

        Args:
            rule_score: Rule engine score (0-100)
            ml_score: ML model score (0-100)

        Returns:
            Confidence score (0-100)
        """
        # Difference between scores
        score_diff = abs(rule_score - ml_score)

        # The smaller the difference, the higher the confidence
        # Max difference 100, min difference 0
        confidence = 100.0 - score_diff

        # Limit to 0-100 range
        confidence = max(0.0, min(100.0, confidence))

        return confidence
