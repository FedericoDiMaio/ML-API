import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List
from logger import logger
from service_intent.intent_ml import IntentClassifier
from service_intent.singleton import Singleton

app = FastAPI()


class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_intent: str
    confidence: float
    risposta: str
    domanda_simile: str
    variabili_coinvolte: List[str] = []
    api_endpoint: str = "unknown"


def get_classifier(json_file_path: str) -> IntentClassifier:
    classifier = Singleton.get_instance(IntentClassifier, json_file_path)

    if not classifier.is_trained:
        logger.info("first API call: Loading and training the model...")
        classifier.load_data()
        classifier.train()
        logger.info("Model training completed!")

    return classifier


@app.post("/predict/ibm", response_model=PredictionResponse)
async def predict_intent(request: PredictionRequest):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, "data", "flussi.json")

    try:
        classifier = get_classifier(json_file_path)
        result = classifier.predict(request.text)

        # Creare una risposta valida secondo il modello PredictionResponse
        response = PredictionResponse(
            predicted_intent=result['predicted_intent'] or "",  # Evita None
            confidence=result['confidence'],
            risposta=result['risposta'] or "",  # Evita None
            domanda_simile=result['domanda_simile'] or "",  # Evita None
            variabili_coinvolte=result['variabili_coinvolte'],
            api_endpoint="ibm"
        )

        return response  # Restituisci la risposta

    except Exception as e:
        logger.error(f"Errore durante la predizione: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )