from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from common import load_pickle, resolve_path
from features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features
from services.weather_api import convert_hourly_to_daily_summary, fetch_erode_weather
from utils.constants import (
    ERODE_TIMEZONE,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)

MODEL_PATH = resolve_path("models", "heatwave_model.pkl")
FEATURE_COLUMNS_PATH = resolve_path("models", "feature_columns.pkl")
TRAINED_META_PATH = resolve_path("models", "training_metadata.pkl")
HISTORICAL_DAILY_CSV = resolve_path("data", "processed", "erode_daily.csv")


@st.cache_resource
def load_ml_model_artifacts():
    """
    Load trained ML model pipeline, feature columns list, and training metadata.
    Uses st.cache_resource for fast memory reuse.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file missing in models/ directory. Run python ai/training/train.py to train.")

    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = load_pickle(FEATURE_COLUMNS_PATH) if FEATURE_COLUMNS_PATH.exists() else []
        metadata = load_pickle(TRAINED_META_PATH) if TRAINED_META_PATH.exists() else {}
        return model, feature_columns, metadata
    except Exception as err:
        # Clear cache in case of broken load state
        st.cache_resource.clear()
        raise RuntimeError(f"Incompatible or corrupted model file: {err}")


def get_historical_daily_history(max_rows: int | None = 60) -> pd.DataFrame:
    """Load recent historical daily weather observations for lag/rolling calculations."""
    if not HISTORICAL_DAILY_CSV.exists():
        raise FileNotFoundError(f"Historical daily file not found at {HISTORICAL_DAILY_CSV}.")
    df = pd.read_csv(HISTORICAL_DAILY_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.tail(max_rows).copy() if max_rows else df.copy()


def calculate_risk_level(probability: float) -> str:
    """Classify probability into LOW, MEDIUM, or HIGH risk level."""
    if probability >= HIGH_RISK_THRESHOLD:
        return RISK_LEVEL_HIGH
    if probability >= MEDIUM_RISK_THRESHOLD:
        return RISK_LEVEL_MEDIUM
    return RISK_LEVEL_LOW


def build_feature_row_from_history(
    combined_df: pd.DataFrame,
    feature_columns: list[str],
    target_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply calendar, lag, and rolling calculations without data leakage.
    Returns (feature_row_df, full_feature_df).
    """
    df = combined_df.copy().sort_values("date").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    latest_row = df[df["date"] == target_date].copy() if target_date is not None else df.tail(1).copy()
    if latest_row.empty:
        raise ValueError(f"No feature row is available for target date {target_date.date() if target_date is not None else target_date}.")
    missing_cols = [c for c in feature_columns if c not in latest_row.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")

    return latest_row[feature_columns], latest_row


def get_realtime_prediction() -> dict[str, object]:
    """
    End-to-end real-time prediction pipeline:
    Historical Daily CSV + Live Weather API -> Leakage-Free Feature Engineering -> Model Inference -> Probability & Risk -> XAI
    """
    try:
        model, feature_columns, metadata = load_ml_model_artifacts()
    except Exception as err:
        return {
            "error": f"Model Service Error: {err}",
            "probability": 0.0,
            "risk_level": "LOW",
            "target_date": "N/A",
            "model_name": "Unavailable",
            "current_weather": fetch_erode_weather()["current"],
        }

    # 1. Fetch live weather & past 7 days hourly data from Open-Meteo
    weather_payload = fetch_erode_weather()
    current_weather = weather_payload["current"]
    hourly_df = weather_payload["hourly_df"]

    # 2. Get baseline daily history
    try:
        history_df = get_historical_daily_history(max_rows=30)
    except Exception as err:
        return {
            "error": f"Historical Data Error: {err}",
            "probability": 0.0,
            "risk_level": "LOW",
            "target_date": "N/A",
            "model_name": metadata.get("best_model", "Trained ML Model"),
            "current_weather": current_weather,
        }

    # 3. Convert live hourly data into daily summary if available, else construct today's row
    if not hourly_df.empty:
        live_daily_df = convert_hourly_to_daily_summary(hourly_df)
        history_dates = set(history_df["date"].dt.date)
        live_daily_df = live_daily_df[~live_daily_df["date"].dt.date.isin(history_dates)]
        combined_df = pd.concat([history_df, live_daily_df], ignore_index=True)
    else:
        today_date = pd.Timestamp.now().floor("D")
        if today_date <= history_df["date"].max():
            today_date = history_df["date"].max() + pd.Timedelta(days=1)

        max_temp = current_weather["temperature_2m"]
        humidity = current_weather["relative_humidity_2m"]
        vpd = current_weather["vapour_pressure_deficit"]

        today_row = {
            "date": today_date,
            "mean_temperature": round(max_temp - 3.0, 1),
            "max_temperature": max_temp,
            "min_temperature": round(max_temp - 8.0, 1),
            "mean_relative_humidity": humidity,
            "max_relative_humidity": min(100.0, humidity + 15.0),
            "mean_wind_speed": current_weather["wind_speed_10m"],
            "max_wind_speed": current_weather["wind_speed_10m"] + 4.0,
            "max_wind_gust": current_weather["wind_gusts_10m"],
            "precipitation_sum": current_weather["precipitation"],
            "mean_vpd": vpd,
            "max_vpd": vpd + 0.5,
            "mean_cloud_cover": current_weather["cloud_cover"],
            "mean_dew_point": current_weather["dew_point_2m"],
            "mean_apparent_temperature": current_weather["apparent_temperature"],
            "max_apparent_temperature": current_weather["apparent_temperature"] + 2.0,
            "min_apparent_temperature": current_weather["apparent_temperature"] - 4.0,
        }
        combined_df = pd.concat([history_df, pd.DataFrame([today_row])], ignore_index=True)

    # 4. Feature engineering & inference
    try:
        feature_row, full_row = build_feature_row_from_history(combined_df, feature_columns)
        pred_class = int(model.predict(feature_row)[0])
        proba = float(model.predict_proba(feature_row)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
        risk_level = calculate_risk_level(proba)
        xai_drivers = compute_feature_contributions(model, feature_row, feature_columns)
        target_date_str = str(full_row["date"].values[0])[:10]

        return {
            "target_date": target_date_str,
            "prediction_class": pred_class,
            "probability": round(proba, 4),
            "risk_level": risk_level,
            "current_weather": current_weather,
            "feature_row": feature_row,
            "full_row": full_row,
            "xai_drivers": xai_drivers,
            "model_name": metadata.get("best_model", "Logistic Regression"),
            "test_metrics": metadata.get("test_metrics", {}),
            "weather_source": weather_payload.get("weather_source", "Open-Meteo current weather API"),
            "weather_error": weather_payload.get("error"),
        }
    except Exception as err:
        return {
            "error": f"Prediction Pipeline Failure: {err}",
            "probability": 0.0,
            "risk_level": "LOW",
            "target_date": "N/A",
            "model_name": metadata.get("best_model", "Trained ML Model"),
            "current_weather": current_weather,
        }


def get_prediction_for_date(target_date: pd.Timestamp | str, weather_payload: dict[str, object] | None = None) -> dict[str, object]:
    """Run the saved heatwave model for one historical or forecast calendar date."""
    target = pd.Timestamp(target_date).normalize()
    today = pd.Timestamp.now(tz=ERODE_TIMEZONE).tz_localize(None).normalize()

    try:
        model, feature_columns, metadata = load_ml_model_artifacts()
        history = get_historical_daily_history(max_rows=None)

        if target > today:
            payload = weather_payload or fetch_erode_weather()
            hourly_df = payload.get("hourly_df", pd.DataFrame())
            forecast_daily = convert_hourly_to_daily_summary(hourly_df)
            target_weather = forecast_daily[forecast_daily["date"] == target].copy()
            if target_weather.empty:
                available_start = forecast_daily["date"].min().date().isoformat() if not forecast_daily.empty else None
                available_end = forecast_daily["date"].max().date().isoformat() if not forecast_daily.empty else None
                return {
                    "available": False,
                    "error": "Requested date is outside the available weather forecast range.",
                    "forecast_range": {"start": available_start, "end": available_end},
                    "target_date": target.date().isoformat(),
                    "weather_source": payload.get("weather_source", "Open-Meteo forecast API"),
                }

            history_before_target = history[history["date"] < target].tail(60)
            history_end = history_before_target["date"].max() if not history_before_target.empty else pd.Timestamp.min
            future_rows = forecast_daily[forecast_daily["date"] > history_end]
            combined_df = pd.concat([history_before_target, future_rows], ignore_index=True)
            weather_source = "Open-Meteo forecast API"
        else:
            target_weather = history[history["date"] == target].copy()
            if target_weather.empty:
                return {
                    "available": False,
                    "error": "Historical weather data is not available for the requested date.",
                    "target_date": target.date().isoformat(),
                    "weather_source": "Historical Erode daily dataset",
                }
            combined_df = history[history["date"] <= target].tail(60).copy()
            weather_source = "Historical Erode daily dataset"

        feature_row, full_row = build_feature_row_from_history(combined_df, feature_columns, target_date=target)
        pred_class = int(model.predict(feature_row)[0])
        probability = float(model.predict_proba(feature_row)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
        target_values = target_weather.iloc[0].to_dict()
        target_values["date"] = target.date().isoformat()
        target_values["temperature_2m"] = float(target_values["max_temperature"])

        return {
            "available": True,
            "target_date": target.date().isoformat(),
            "date_type": "future" if target > today else "past",
            "weather": target_values,
            "temperature": float(target_values["temperature_2m"]),
            "prediction_class": pred_class,
            "probability": round(probability, 4),
            "risk_level": calculate_risk_level(probability),
            "feature_row": feature_row,
            "full_row": full_row,
            "xai_drivers": compute_feature_contributions(model, feature_row, feature_columns),
            "model_name": metadata.get("best_model", "Trained ML Model"),
            "weather_source": weather_source,
            "forecast_range": payload.get("forecast_range") if target > today else None,
        }
    except Exception as err:
        return {
            "available": False,
            "error": f"Date-specific prediction failed: {err}",
            "target_date": target.date().isoformat(),
            "weather_source": "Unavailable",
        }


def compute_feature_contributions(model, feature_row: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """
    Extract actual model feature importances and combine with instance feature values to rank key drivers.
    """
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        importances = np.ones(len(feature_columns)) / len(feature_columns)

    row_vals = feature_row.iloc[0].values
    contrib_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importances,
            "value": row_vals,
            "score": importances * np.abs(row_vals),
        }
    )

    contrib_df = contrib_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return contrib_df


def predict_custom_scenario(
    max_temp: float,
    mean_temp: float,
    humidity: float,
    vpd: float,
    wind_speed: float,
) -> dict[str, object]:
    """
    What-If scenario testing: override current weather parameters with custom values
    and compute updated rolling features and prediction probability.
    """
    model, feature_columns, _ = load_ml_model_artifacts()
    history_df = get_historical_daily_history(max_rows=30)

    target_date = history_df["date"].max() + pd.Timedelta(days=1)
    scenario_row = {
        "date": target_date,
        "mean_temperature": mean_temp,
        "max_temperature": max_temp,
        "min_temperature": round(max_temp - 7.0, 1),
        "mean_relative_humidity": humidity,
        "max_relative_humidity": min(100.0, humidity + 15.0),
        "mean_wind_speed": wind_speed,
        "max_wind_speed": wind_speed + 5.0,
        "max_wind_gust": wind_speed + 10.0,
        "precipitation_sum": 0.0,
        "mean_vpd": vpd,
        "max_vpd": vpd + 0.4,
        "mean_cloud_cover": 20.0,
        "mean_dew_point": round(mean_temp - ((100.0 - humidity) / 5.0), 1),
        "mean_apparent_temperature": mean_temp + 2.0,
        "max_apparent_temperature": max_temp + 3.0,
        "min_apparent_temperature": round(max_temp - 5.0, 1),
    }

    combined_df = pd.concat([history_df, pd.DataFrame([scenario_row])], ignore_index=True)
    feature_row, full_row = build_feature_row_from_history(combined_df, feature_columns)

    pred_class = int(model.predict(feature_row)[0])
    proba = float(model.predict_proba(feature_row)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
    risk_level = calculate_risk_level(proba)
    xai_drivers = compute_feature_contributions(model, feature_row, feature_columns)

    return {
        "target_date": str(target_date)[:10],
        "prediction_class": pred_class,
        "probability": round(proba, 4),
        "risk_level": risk_level,
        "feature_row": feature_row,
        "xai_drivers": xai_drivers,
    }
