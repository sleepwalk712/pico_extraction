from typing import List, Optional

from pydantic import BaseModel


class ClassificationFineTuneRequest(BaseModel):
    ml_model_path: str
    texts: List[str]
    labels: List[int]
    epochs: int = 3
    num_labels: Optional[int] = 2
