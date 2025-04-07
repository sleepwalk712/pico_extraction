from fastapi import APIRouter, HTTPException

from app.services.ner_service import NERService
from app.models.schemas import PredictRequest, FineTuneRequest

router = APIRouter()


@router.post("/predict", response_model=dict[str, list[int]])
async def predict(request: PredictRequest) -> dict[str, list[int]]:
    try:
        if not request.ml_model_path:
            raise HTTPException(
                status_code=400, detail="ml_model_path is required"
            )

        ner_service = NERService(model_path=request.ml_model_path)
        predictions = ner_service.predict(request.text)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fine_tune", response_model=dict[str, str])
async def fine_tune(request: FineTuneRequest) -> dict[str, str]:
    try:
        if not request.ml_model_path:
            raise HTTPException(
                status_code=400, detail="ml_model_path is required"
            )

        ner_service = NERService(model_path=request.ml_model_path)
        ner_service.fine_tune(
            texts=request.texts,
            labels=request.labels,
            epochs=request.epochs,
        )
        return {"message": "Fine-tuning completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
