from typing import Optional

from pydantic import BaseModel


class ClassificationPredictRequest(BaseModel):
    ml_model_path: str
    text: str
    num_labels: Optional[int] = 2
