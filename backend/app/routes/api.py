import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.database.store import store
from app.ml.pipeline import FEATURES, REQUIRED_COLUMNS, TARGETS, load_model, train_models
from app.models.schemas import PredictionRequest
from app.services.weather import prediction_features, synthetic_weather_data, uncertainty_from_residuals

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
_active_model: dict | None = None
_uploaded_data: pd.DataFrame | None = None
PREDICTION_BOUNDS = {
    "temperature": (-90.0, 65.0),
    "humidity": (0.0, 100.0),
    "rainfall": (0.0, 1000.0),
    "wind_speed": (0.0, 500.0),
}


def constrain_predictions(values: list[float]) -> dict[str, float]:
    return {
        target: round(
            min(max(float(value), PREDICTION_BOUNDS[target][0]), PREDICTION_BOUNDS[target][1]),
            2,
        )
        for target, value in zip(TARGETS, values)
    }


def get_active_model() -> dict:
    global _active_model
    if _active_model is not None:
        return _active_model
    latest = store.latest()
    if latest and Path(latest["model_file"]).exists():
        _active_model = load_model(latest["model_file"])
        return _active_model
    return {}


def train_and_store(frame: pd.DataFrame) -> dict:
    latest = store.latest()
    record = train_models(frame, get_settings().resolved_model_path, latest["iteration"] if latest else 0)
    store.save_iteration(record)
    global _active_model
    _active_model = load_model(record["model_file"])
    return record


@router.get("/health")
def health():
    return {"status": "ok", "database": "mongodb" if get_settings().mongodb_uri else "in-memory", "model_loaded": bool(get_active_model())}


@router.get("/features")
def features():
    return {"features": FEATURES, "targets": TARGETS, "required_csv_columns": REQUIRED_COLUMNS}


@router.post("/predict")
def predict(request: PredictionRequest):
    model = get_active_model()
    if not model:
        raise HTTPException(status_code=503, detail="No trained model is available. Upload a dataset or train the sample model first.")
    record = store.latest()
    row = prediction_features(request.model_dump())
    values = model["pipeline"].predict(row)[0]
    predictions = constrain_predictions(values.tolist())
    uncertainty = uncertainty_from_residuals(record.get("residuals", [])) if record else {}
    response = {
        "city": request.city, "latitude": request.latitude, "longitude": request.longitude,
        "predicted_at": datetime.now(timezone.utc).isoformat(), "predictions": predictions,
        "uncertainty": uncertainty, "model_name": model["model_name"], "version": model["version"],
    }
    store.save_prediction(response)
    return response


@router.post("/train")
def train():
    global _uploaded_data
    try:
        record = train_and_store(_uploaded_data if _uploaded_data is not None else synthetic_weather_data())
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.exception("Model training failed")
        raise HTTPException(status_code=500, detail="Model training failed.") from error
    return {"message": "Training completed", "model": record}


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    global _uploaded_data
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Upload a .csv file.")
    max_bytes = get_settings().max_upload_mb * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(status_code=413, detail=f"CSV exceeds the {get_settings().max_upload_mb} MB limit.")
    try:
        frame = pd.read_csv(pd.io.common.BytesIO(contents))
        from app.ml.pipeline import prepare_data
        _, _, stats = prepare_data(frame)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    _uploaded_data = frame
    return {"message": "Dataset validated and staged for training", "stats": stats}


@router.get("/model/latest")
def latest_model():
    record = store.latest()
    if record is None:
        return {"model": None}
    return {"model": {key: value for key, value in record.items() if key != "model_file"}}


@router.get("/model/history")
def model_history():
    return {"iterations": [{key: value for key, value in row.items() if key not in ("model_file", "residuals", "actual_vs_predicted", "dataset_stats")} for row in store.history()]}


@router.get("/model/comparison")
def model_comparison():
    record = store.latest()
    return {"models": record["comparison"] if record else []}


@router.get("/weather/history")
def weather_history():
    return {"predictions": store.predictions()}
