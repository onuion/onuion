"""
onuion - Real-time Security Risk Analysis Model

Open-source hybrid (rule-based + ML) security risk analysis system
for session data analysis with sub-millisecond inference time.
"""

__version__ = "0.1.0"
__author__ = "onuion contributors"

from onuion.inference import analyze_risk, RiskAnalysisResult
from onuion.model import RiskModel

# Optional Hugging Face integration
try:
    from onuion.huggingface import (
        HuggingFaceIntegration,
        upload_to_hub,
        download_from_hub
    )
    __all__ = [
        "analyze_risk",
        "RiskAnalysisResult",
        "RiskModel",
        "HuggingFaceIntegration",
        "upload_to_hub",
        "download_from_hub",
    ]
except ImportError:
    __all__ = [
        "analyze_risk",
        "RiskAnalysisResult",
        "RiskModel",
    ]
