from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.ml.pipeline import FEATURES


def synthetic_weather_data(rows: int = 240) -> pd.DataFrame:
    from pathlib import Path
    possible_paths = [
        Path("backend/datasets/sample_weather.csv"),
        Path("datasets/sample_weather.csv"),
        Path.cwd() / "backend" / "datasets" / "sample_weather.csv",
    ]
    for csv_path in possible_paths:
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if len(df) >= 12:
                    if rows and rows < len(df):
                        return df.tail(rows).reset_index(drop=True)
                    return df
            except Exception:
                pass

    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h"), periods=rows, freq="h")
    hours = np.arange(rows)
    daily = np.sin(2 * np.pi * (hours % 24) / 24)
    seasonal = np.sin(2 * np.pi * hours / (24 * 30))
    temperature = 26 + 3.2 * daily + 2.8 * seasonal + rng.normal(0, 0.8, rows)
    humidity = np.clip(70 - 10 * daily - 8 * seasonal + rng.normal(0, 4, rows), 35, 98)
    rainfall = np.maximum(0, rng.gamma(0.7, 1.4, rows) * (humidity > 72))
    wind = np.clip(12 + 3 * np.cos(hours / 19) + rng.normal(0, 2, rows), 2, 35)
    pressure = 1012 + 4 * np.sin(hours / 29) + rng.normal(0, 1.2, rows)
    return pd.DataFrame({
        "date": dates, "latitude": 19.076, "longitude": 72.878,
        "temperature": temperature, "humidity": humidity, "rainfall": rainfall,
        "wind_speed": wind, "pressure": pressure,
    })


def prediction_features(request: dict, history: pd.DataFrame | None = None) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    values = {
        "latitude": request["latitude"], "longitude": request["longitude"],
        "temperature": request["temperature"], "humidity": request["humidity"],
        "wind_speed": request["wind_speed"], "pressure": request["pressure"], "rainfall": request["rainfall"],
        "month": now.month, "day": now.day, "hour": now.hour,
        "previous_temperature": request["temperature"], "previous_humidity": request["humidity"],
        "previous_rainfall": request["rainfall"], "rolling_temperature": request["temperature"],
        "rolling_humidity": request["humidity"], "rolling_rainfall": request["rainfall"],
    }
    if history is not None and not history.empty:
        for col in ["temperature", "humidity", "rainfall"]:
            if col in history:
                values[f"previous_{col}"] = float(history[col].iloc[-1])
                values[f"rolling_{col}"] = float(history[col].tail(6).mean())
    return pd.DataFrame([{feature: values[feature] for feature in FEATURES}])


def uncertainty_from_residuals(residuals: list[dict]) -> dict[str, float]:
    uncertainty = {}
    for target in ("temperature", "humidity", "rainfall", "wind_speed"):
        values = [abs(row["residual"]) for row in residuals if row["target"] == target]
        uncertainty[target] = round(float(np.quantile(values, 0.9)), 2) if values else 0.0
    return uncertainty
