from pydantic import BaseModel
from typing import List


class ECGRequest(BaseModel):
    sequence: List[float]


class PredictionResponse(BaseModel):
    anomaly: bool
    score: float
    threshold: float
