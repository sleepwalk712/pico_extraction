from typing import List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str
