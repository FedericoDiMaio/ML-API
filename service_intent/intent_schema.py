from typing import Optional

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    # predicted_intent: str
    # confidence: float
    # similar_examples: List[SimilarExample]
    api_endpoint: Optional[str] = None


class PredictionRequest(BaseModel):
    text: str
    # threshold: Optional[float] = 0.3
