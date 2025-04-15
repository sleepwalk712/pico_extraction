from pydantic import BaseModel


class NerPredictRequest(BaseModel):
    text: str
    ml_model_path: str
