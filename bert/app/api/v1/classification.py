from fastapi import APIRouter, HTTPException

from app.services.cls_service import ClassificationService
from app.models.schemas import ClassificationPredictRequest, ClassificationFineTuneRequest

router = APIRouter()


@router.post("/classification/predict", response_model=dict[str, int])
async def classification_predict(request: ClassificationPredictRequest) -> dict[str, int]:
    try:
        if not request.ml_model_path:
            raise HTTPException(
                status_code=400, detail="ml_model_path is required")

        cls_service = ClassificationService(
            model_path=request.ml_model_path,
            num_labels=request.num_labels or 2
        )
        prediction = cls_service.predict(request.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/classification/fine_tune", response_model=dict[str, str])
async def classification_fine_tune(request: ClassificationFineTuneRequest) -> dict[str, str]:
    try:
        if not request.ml_model_path:
            raise HTTPException(
                status_code=400, detail="ml_model_path is required")

        cls_service = ClassificationService(
            model_path=request.ml_model_path,
            num_labels=request.num_labels or 2
        )
        cls_service.fine_tune(
            texts=request.texts,
            labels=request.labels,
            epochs=request.epochs,
        )
        return {"message": "Fine-tuning completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
