from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    temperature: float = Field(default=25, ge=-90, le=65)
    humidity: float = Field(default=65, ge=0, le=100)
    rainfall: float = Field(default=0, ge=0, le=1000)
    wind_speed: float = Field(default=10, ge=0, le=500)
    pressure: float = Field(default=1013, ge=800, le=1200)


class PredictionResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    predicted_at: datetime
    predictions: dict[str, float]
    uncertainty: dict[str, float]
    model_name: str
    version: str

