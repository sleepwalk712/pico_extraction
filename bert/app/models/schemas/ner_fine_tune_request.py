from typing import List

from pydantic import BaseModel


class NerFineTuneRequest(BaseModel):
    texts: List[List[str]]
    labels: List[List[int]]
    epochs: int = 3
    ml_model_path: str
