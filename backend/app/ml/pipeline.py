import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

TARGETS = ["temperature", "humidity", "rainfall", "wind_speed"]
REQUIRED_COLUMNS = ["date", "latitude", "longitude", "temperature", "humidity", "rainfall", "wind_speed", "pressure"]
FEATURES = [
    "latitude", "longitude", "temperature", "humidity", "wind_speed", "pressure", "rainfall",
    "month", "day", "hour", "previous_temperature", "previous_humidity", "previous_rainfall",
    "rolling_temperature", "rolling_humidity", "rolling_rainfall",
]


def prepare_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")
    if len(frame) < 12:
        raise ValueError("At least 12 valid weather records are required for training.")

    data = frame[REQUIRED_COLUMNS].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "latitude", "longitude"]).sort_values("date").reset_index(drop=True)
    if len(data) < 12:
        raise ValueError("At least 12 records with valid dates and coordinates are required.")
    if ((data.latitude < -90) | (data.latitude > 90) | (data.longitude < -180) | (data.longitude > 180)).any():
        raise ValueError("Latitude or longitude is outside its valid range.")
    missing_values = int(data[REQUIRED_COLUMNS].isna().sum().sum())
    duplicate_count = int(data.duplicated(subset=["date", "latitude", "longitude"]).sum())
    data = data.drop_duplicates(subset=["date", "latitude", "longitude"], keep="last").reset_index(drop=True)
    if len(data) < 12:
        raise ValueError("At least 12 unique weather records are required after duplicate removal.")

    bounds = {
        "temperature": (-90, 65), "humidity": (0, 100), "rainfall": (0, 1000),
        "wind_speed": (0, 500), "pressure": (800, 1200),
    }
    outlier_counts: dict[str, int] = {}
    for column, (low, high) in bounds.items():
        outlier_counts[column] = int(((data[column] < low) | (data[column] > high)).sum())
        data.loc[(data[column] < low) | (data[column] > high), column] = np.nan

    numeric = data[REQUIRED_COLUMNS[1:]].columns
    data[numeric] = data[numeric].ffill()
    dates = data["date"]
    features = data[["latitude", "longitude", "temperature", "humidity", "wind_speed", "pressure", "rainfall"]].copy()
    features["month"] = dates.dt.month
    features["day"] = dates.dt.day
    features["hour"] = dates.dt.hour
    for source, lag in [("temperature", "previous_temperature"), ("humidity", "previous_humidity"), ("rainfall", "previous_rainfall")]:
        features[lag] = data[source].shift(24).fillna(data[source])
    features["rolling_temperature"] = data.temperature.shift(1).rolling(6, min_periods=1).mean().fillna(data.temperature)
    features["rolling_humidity"] = data.humidity.shift(1).rolling(6, min_periods=1).mean().fillna(data.humidity)
    features["rolling_rainfall"] = data.rainfall.shift(1).rolling(6, min_periods=1).mean().fillna(data.rainfall)
    targets = data[TARGETS].shift(-1).reset_index(drop=True)
    features = features[FEATURES].replace([np.inf, -np.inf], np.nan)
    valid_targets = targets.notna().all(axis=1)
    features = features.loc[valid_targets].reset_index(drop=True)
    targets = targets.loc[valid_targets].reset_index(drop=True)
    if len(features) < 8:
        raise ValueError("At least 8 consecutive weather observations with valid targets are required.")
    stats = {
        "records": int(len(data)),
        "missing_values": missing_values,
        "duplicate_records": duplicate_count,
        "outliers": outlier_counts,
        "columns": [{"name": c, "dtype": str(data[c].dtype)} for c in REQUIRED_COLUMNS],
        "statistics": data[TARGETS + ["pressure"]].describe().round(2).replace({np.nan: None}).to_dict(),
        "preview": data.head(8).assign(date=lambda x: x.date.astype(str)).to_dict(orient="records"),
    }
    return features, targets, stats


def train_models(frame: pd.DataFrame, model_dir: Path, previous_iterations: int = 0) -> dict[str, Any]:
    X, y, dataset_stats = prepare_data(frame)
    split_at = max(2, min(len(X) - 1, int(len(X) * 0.8)))
    X_train, X_test = X.iloc[:split_at], X.iloc[split_at:]
    y_train, y_test = y.iloc[:split_at], y.iloc[split_at:]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, FEATURES)])
    candidates: list[tuple[str, Any]] = [
        ("Linear Regression", LinearRegression()),
        ("Random Forest", MultiOutputRegressor(RandomForestRegressor(n_estimators=120, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1))),
        ("Gradient Boosting", MultiOutputRegressor(GradientBoostingRegressor(random_state=42))),
    ]
    comparisons = []
    fitted: dict[str, Any] = {}
    cv = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train) // 3)))
    for name, estimator in candidates:
        start = time.perf_counter()
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        mae = float(mean_absolute_error(y_test, predictions))
        r2 = float(r2_score(y_test, predictions, multioutput="uniform_average"))
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2", n_jobs=1)
        comparisons.append({
            "model": name, "mae": mae, "rmse": rmse, "r2": r2,
            "training_time": round(time.perf_counter() - start, 3),
            "cv_r2": float(np.mean(cv_scores)),
        })
        fitted[name] = (pipeline, predictions)

    best_row = max(comparisons, key=lambda item: item["r2"])
    best_name = best_row["model"]
    best_pipeline, best_predictions = fitted[best_name]
    # Small, bounded tuning step on the selected family; retained only when it improves held-out R².
    try:
        if best_name == "Random Forest":
            search = GridSearchCV(
                Pipeline([("preprocessor", preprocessor), ("regressor", MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1)))]),
                {"regressor__estimator__n_estimators": [80, 140], "regressor__estimator__max_depth": [8, 14]},
                cv=cv, scoring="r2", n_jobs=1,
            )
            search.fit(X_train, y_train)
            tuned_predictions = search.best_estimator_.predict(X_test)
            if r2_score(y_test, tuned_predictions) > best_row["r2"]:
                best_pipeline, best_predictions = search.best_estimator_, tuned_predictions
    except (ValueError, TypeError):
        logger.exception("Hyperparameter search failed; using the best baseline estimator")

    best_row["mae"] = float(mean_absolute_error(y_test, best_predictions))
    best_row["rmse"] = float(np.sqrt(mean_squared_error(y_test, best_predictions)))
    best_row["r2"] = float(r2_score(y_test, best_predictions, multioutput="uniform_average"))
    r2_by_target = {target: float(r2_score(y_test[target], best_predictions[:, i])) for i, target in enumerate(TARGETS)}
    iteration = previous_iterations + 1
    version = f"v{iteration}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"weather_model_{version}.joblib"
    joblib.dump({"pipeline": best_pipeline, "features": FEATURES, "targets": TARGETS, "version": version, "model_name": best_name}, model_file)

    feature_importance: list[dict[str, Any]] = []
    try:
        model = best_pipeline.named_steps["regressor"]
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0) if isinstance(model, MultiOutputRegressor) else np.abs(model.coef_).mean(axis=0)
        feature_importance = [{"feature": feature, "importance": float(value)} for feature, value in sorted(zip(FEATURES, importances), key=lambda pair: pair[1], reverse=True)]
    except (AttributeError, ValueError):
        logger.info("Feature importance is unavailable for %s", best_name)

    actual_vs_predicted = [
        {"actual": float(a), "predicted": float(p), "target": target}
        for i, target in enumerate(TARGETS)
        for a, p in zip(y_test[target].tolist()[:40], best_predictions[:, i].tolist()[:40])
    ]
    residuals = [{"actual": float(a), "predicted": float(p), "residual": float(a - p), "target": target}
                 for i, target in enumerate(TARGETS) for a, p in zip(y_test[target].tolist()[:40], best_predictions[:, i].tolist()[:40])]
    record = {
        "iteration": iteration, "version": version, "model_name": best_name,
        "training_date": datetime.now(timezone.utc).isoformat(), "dataset_size": int(len(X)),
        "features": len(FEATURES), "mae": best_row["mae"], "rmse": best_row["rmse"],
        "r2_score": best_row["r2"], "r2_by_target": r2_by_target,
        "mse": float(mean_squared_error(y_test, best_predictions)),
        "mape": {target: float(np.mean(np.abs((y_test[target] - best_predictions[:, i]) / np.maximum(np.abs(y_test[target]), 1))) * 100) for i, target in enumerate(TARGETS)},
        "comparison": comparisons, "feature_importance": feature_importance,
        "actual_vs_predicted": actual_vs_predicted, "residuals": residuals,
        "dataset_stats": dataset_stats, "model_file": str(model_file),
    }
    return record


def load_model(path: str) -> dict[str, Any]:
    return joblib.load(path)
