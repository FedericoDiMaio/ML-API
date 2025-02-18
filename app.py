import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from service.intent_ml import get_classifier

app = FastAPI(title="ML - API")

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, "data")
train_dir = os.path.join(dataset_dir, "train_intent")

classifier = get_classifier(train_dir)


class SimilarExample(BaseModel):
    text: str
    intent: str
    similarity: float


class PredictionResponse(BaseModel):
    predicted_intent: str
    confidence: float
    similar_examples: List[SimilarExample]
    api_endpoint: Optional[str] = None


class PredictionRequest(BaseModel):
    text: str
    threshold: Optional[float] = 0.3


@app.post("/predict", response_model=PredictionResponse)
async def predict_intent(request: PredictionRequest):
    try:
        # Get prediction
        result = classifier.predict(request.text, request.threshold)

        # Map intent to API endpoint
        intent_to_api = {
            "get_weather": "/api/weather",
            "get_time": "/api/time",
            "get_news": "/api/news",
        }

        # Add API endpoint to response
        result['api_endpoint'] = intent_to_api.get(result['predicted_intent'], "unknown")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
