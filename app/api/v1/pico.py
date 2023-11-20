from fastapi import APIRouter, HTTPException

from app.services.ner_service import NERService
from app.models.schemas import PredictRequest
from app.models.schemas import FineTuneRequest


router = APIRouter()
ner_service = NERService()


@router.post("/predict")
async def predict(request: PredictRequest):
    try:
        predictions = ner_service.predict(request.text)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine_tune")
async def fine_tune(request: FineTuneRequest):
    try:
        ner_service.fine_tune(
            request.texts,
            request.labels,
            epochs=request.epochs,
            ml_model_path=request.ml_model_path
        )
        return {"message": "Fine-tuning completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
