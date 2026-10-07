import pandas as pd
import pytest

from app.ml.pipeline import prepare_data
from app.services.weather import synthetic_weather_data


def test_feature_engineering_returns_all_required_features():
    features, targets, stats = prepare_data(synthetic_weather_data(48))
    assert len(features) == len(targets) == 47
    assert {"previous_temperature", "rolling_rainfall", "hour"} <= set(features.columns)
    assert set(targets.columns) == {"temperature", "humidity", "rainfall", "wind_speed"}
    assert stats["records"] == 48


def test_training_compares_models_and_versions_artifact(tmp_path):
    from app.ml.pipeline import train_models

    result = train_models(synthetic_weather_data(48), tmp_path)
    assert result["version"] == "v1"
    assert result["model_name"] in {"Linear Regression", "Random Forest", "Gradient Boosting"}
    assert {row["model"] for row in result["comparison"]} == {"Linear Regression", "Random Forest", "Gradient Boosting"}
    assert result["dataset_size"] == 47
    assert (tmp_path / "weather_model_v1.joblib").exists()


def test_missing_required_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_data(pd.DataFrame({"date": ["2025-01-01"]}))


def test_invalid_coordinates_are_rejected():
    frame = synthetic_weather_data(24)
    frame.loc[0, "latitude"] = 95
    with pytest.raises(ValueError, match="outside its valid range"):
        prepare_data(frame)


def test_predictions_are_clamped_to_physical_bounds():
    from app.routes.api import constrain_predictions

    assert constrain_predictions([25, 110, -4, 12]) == {
        "temperature": 25,
        "humidity": 100,
        "rainfall": 0,
        "wind_speed": 12,
    }
