"""
Inference Pipeline Module

Performs end-to-end risk analysis by combining all components.
Optimized for real-time inference.
"""

import time
from typing import Dict, Any, Optional
import numpy as np

from onuion.feature_extractor import FeatureExtractor
from onuion.rule_engine import RuleEngine
from onuion.model import RiskModel
from onuion.risk_aggregator import RiskAggregator


class RiskAnalysisResult:
    """Data class for risk analysis result."""
    
    def __init__(
        self,
        risk_score: float,
        risk: list,
        rule_score: float,
        ml_score: float,
        confidence: float,
        inference_time_ms: float
    ):
        self.riskScore = risk_score
        self.risk = risk
        self.rule_score = rule_score
        self.ml_score = ml_score
        self.confidence = confidence
        self.inference_time_ms = inference_time_ms
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary format."""
        return {
            "riskScore": self.riskScore,
            "risk": self.risk,
            "rule_score": self.rule_score,
            "ml_score": self.ml_score,
            "confidence": self.confidence,
            "inference_time_ms": round(self.inference_time_ms, 3),
        }


class InferencePipeline:
    """
    End-to-end inference pipeline.
    
    Pipeline:
    1. Feature extraction (< 0.1ms)
    2. Rule engine evaluation (< 0.1ms)
    3. ML model prediction (< 0.5ms)
    4. Risk aggregation (< 0.01ms)
    
    Total target: < 1ms
    """
    
    def __init__(
        self,
        model: Optional[RiskModel] = None,
        model_path: Optional[str] = None
    ):
        """
        Initializes the inference pipeline.
        
        Args:
            model: RiskModel instance (optional)
            model_path: SavedModel path (optional, used if model not provided)
        """
        self.feature_extractor = FeatureExtractor()
        self.rule_engine = RuleEngine()
        self.risk_aggregator = RiskAggregator()
        
        # Model loading
        if model is not None:
            self.model = model
        elif model_path is not None:
            self.model = RiskModel()
            self.model.load(model_path)
        else:
            # Default model (untrained, for inference only)
            # Production must load a trained model
            self.model = RiskModel()
            print("WARNING: Using untrained default model!")
    
    def analyze(self, session_data: Dict[str, Any]) -> RiskAnalysisResult:
        """
        Analyzes session data and produces risk score.
        
        Args:
            session_data: Session data to analyze
            
        Returns:
            RiskAnalysisResult object
        """
        start_time = time.perf_counter()
        
        # 1. Feature extraction
        features = self.feature_extractor.extract(session_data)
        
        # 2. Rule engine evaluation
        rule_result = self.rule_engine.evaluate(session_data)
        
        # 3. ML model prediction
        ml_score = self.model.predict(features)
        
        # 4. Risk aggregation
        aggregated = self.risk_aggregator.aggregate(rule_result, ml_score)
        
        # Inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ms
        
        return RiskAnalysisResult(
            risk_score=aggregated["riskScore"],
            risk=aggregated["risk"],
            rule_score=aggregated["rule_score"],
            ml_score=aggregated["ml_score"],
            confidence=aggregated["confidence"],
            inference_time_ms=inference_time
        )
    
    def analyze_batch(
        self,
        session_data_list: list
    ) -> list:
        """
        Performs batch inference (more efficient).
        
        Args:
            session_data_list: List of session data
            
        Returns:
            List of RiskAnalysisResult
        """
        results = []
        
        # Feature extraction (batch)
        features_list = [
            self.feature_extractor.extract(sd) for sd in session_data_list
        ]
        features_batch = np.array(features_list)
        
        # Rule evaluation (for each)
        rule_results = [
            self.rule_engine.evaluate(sd) for sd in session_data_list
        ]
        
        # ML prediction (batch)
        ml_scores = self.model.predict_batch(features_batch)
        
        # Aggregation
        for i, (rule_result, ml_score) in enumerate(zip(rule_results, ml_scores)):
            aggregated = self.risk_aggregator.aggregate(rule_result, float(ml_score))
            results.append(RiskAnalysisResult(
                risk_score=aggregated["riskScore"],
                risk=aggregated["risk"],
                rule_score=aggregated["rule_score"],
                ml_score=aggregated["ml_score"],
                confidence=aggregated["confidence"],
                inference_time_ms=0.0  # Total time should be measured in batch
            ))
        
        return results


# Global pipeline instance (lazy loading)
_pipeline: Optional[InferencePipeline] = None


def get_pipeline(model_path: Optional[str] = None) -> InferencePipeline:
    """
    Returns global pipeline instance (singleton pattern).
    
    Args:
        model_path: Model path (used on first call)
        
    Returns:
        InferencePipeline instance
    """
    global _pipeline
    
    if _pipeline is None:
        _pipeline = InferencePipeline(model_path=model_path)
    
    return _pipeline


def analyze_risk(
    session_data: Dict[str, Any],
    model_path: Optional[str] = None
) -> RiskAnalysisResult:
    """
    Convenience function: Analyzes session data.
    
    Args:
        session_data: Session data to analyze
        model_path: Model path (optional, used on first call)
        
    Returns:
        RiskAnalysisResult object
    """
    pipeline = get_pipeline(model_path=model_path)
    return pipeline.analyze(session_data)

