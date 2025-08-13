from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    """
    Schema for the input data for a prediction request.
    """
    data: List[float]

class PredictionOutput(BaseModel):
    """
    Schema for the output of a prediction request.
    """
    forecast: List[float]
