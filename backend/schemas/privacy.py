from pydantic import BaseModel
from typing import Dict, Any

class PrivacyStatus(BaseModel):
    differentialPrivacy: Dict[str, Any]
    epsilonBudget: float
    secureAggregation: Dict[str, Any]
    securityScore: float
