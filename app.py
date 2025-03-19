import os

import uvicorn
from fastapi import FastAPI, HTTPException

from service_intent.intent_manager import get_classifier
from service_intent.intent_schema import PredictionResponse, PredictionRequest

app = FastAPI(title="ML - API")


@app.post("/predict", response_model=PredictionResponse)
async def predict_intent(request: PredictionRequest):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(current_dir, "data1")
    train_dir = os.path.join(dataset_dir, "train_intent")

    try:
        classifier = get_classifier(train_dir)
        result = classifier.predict(request.text)

        intent_to_api = {
            "getAuleLibereUtente": "/Aule/getAuleLibereUtente",
            "getImpegniByDocente": "/Impegni/getImpegniByDocente",
            "orarioLezioniDocente": "/Impegni/orarioLezioniDocente",
        }

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
