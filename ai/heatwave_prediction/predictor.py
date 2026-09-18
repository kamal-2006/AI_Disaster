from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

def calculate_risk_level(probability: float) -> str:
    """Classify probability into LOW, MEDIUM/MODERATE, or HIGH risk level."""
    if probability >= 0.80:
        return "HIGH"
    if probability >= 0.50:
        return "MEDIUM"
    return "LOW"

from .config import HISTORICAL_DAILY_PATH, LOCATION, MODEL_PATH
from .date_parser import format_display_date, get_current_today, parse_date_input
from .preprocessing import build_model_features, build_climatological_features
from .weather_service import fetch_weather_for_date


def format_user_response(result: Dict[str, Any]) -> str:
    """Format final output string matching requirement #20 format."""
    if not result.get("available"):
        return result.get("explanation") or result.get("error") or "Heatwave prediction is unavailable for the requested date."

    date_type = result.get("date_type", "").lower()
    location = result.get("location", LOCATION)
    formatted_date = result.get("formatted_date", result.get("date"))
    temperature = result.get("temperature")
    temp_str = f"{float(temperature):.1f}°C" if temperature is not None else "N/A"
    risk = result.get("risk", "Low").title()
    pred_class = result.get("prediction_class", 0)
    pred_source = result.get("prediction_source", "forecast")

    if date_type == "today":
        if pred_class == 1 or risk.upper() in {"HIGH", "MEDIUM", "MODERATE"}:
            pred_desc = f"Current weather conditions indicate a {risk.lower()} heatwave risk."
        else:
            pred_desc = "Current weather conditions indicate a low heatwave risk."

        return (
            "## Heatwave Prediction\n\n"
            f"Location: {location}\n"
            "Date: Today\n\n"
            f"Current Temperature: {temp_str}\n"
            f"Heatwave Risk: {risk}\n\n"
            f"Prediction: {pred_desc}"
        )

    elif date_type == "future":
        source_label = "Historical Climatology + SafeGraph AI ML Model" if pred_source == "historical_model" else "Open-Meteo Forecast + SafeGraph AI Heatwave Model"
        if pred_class == 1 or risk.upper() == "HIGH":
            pred_desc = "Heatwave conditions are likely based on the weather features and trained ML model."
        elif risk.upper() in {"MEDIUM", "MODERATE"}:
            pred_desc = "Moderate heatwave conditions are possible based on the weather features and trained ML model."
        else:
            pred_desc = "Heatwave conditions are unlikely based on the weather features and trained ML model."

        return (
            "## Heatwave Prediction\n\n"
            f"Location: {location}\n"
            f"Date: {formatted_date}\n\n"
            f"Forecast Temperature: {temp_str}\n"
            f"Heatwave Risk: {risk}\n\n"
            f"Prediction: {pred_desc}\n\n"
            f"Data Source:\n{source_label}"
        )

    else:
        # Historical / Past
        return (
            "## Heatwave Prediction (Historical)\n\n"
            f"Location: {location}\n"
            f"Date: {formatted_date}\n\n"
            f"Recorded Temperature: {temp_str}\n"
            f"Heatwave Risk: {risk}\n\n"
            f"Prediction: Historical weather records show a {risk.lower()} heatwave risk level."
        )


def _log_debug_info(parsed: Dict[str, Any], weather: Dict[str, Any], feature_row: pd.DataFrame | None, pred_class: int | None, risk: str | None) -> None:
    """Print debug output matching requirement #17 when SAFEGRAPH_AI_DEBUG is set."""
    if os.getenv("SAFEGRAPH_AI_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return

    feat_dict = feature_row.iloc[0].to_dict() if feature_row is not None and not feature_row.empty else {}
    print("\n---")
    print("## HEATWAVE PREDICTION DEBUG")
    print(f"User Input: {parsed.get('raw_input')}")
    print(f"Parsed Date: {parsed.get('iso_date')}")
    print(f"Date Type: {parsed.get('date_type')}")
    print(f"Location: {LOCATION}")
    print(f"Weather API URL: {weather.get('api_url', 'N/A')}")
    print(f"Forecast Date Retrieved: {weather.get('target_date', 'N/A')}")
    print(f"Temperature: {weather.get('target_weather', {}).get('max_temperature', weather.get('current', {}).get('temperature_2m'))}")
    print(f"Humidity: {weather.get('target_weather', {}).get('mean_relative_humidity', weather.get('current', {}).get('relative_humidity_2m'))}")
    print(f"Wind Speed: {weather.get('target_weather', {}).get('mean_wind_speed', weather.get('current', {}).get('wind_speed_10m'))}")
    print(f"Other Model Features: {list(feat_dict.keys())[:5]}... (Total: {len(feat_dict)})")
    print(f"Processed Features: {feat_dict}")
    print(f"Model Prediction: {pred_class}")
    print(f"Risk: {risk}")
    print("-----\n")


def _failure_result(target_date_str: str, formatted_date_str: str, message: str, source: str, date_type: str = "future") -> Dict[str, Any]:
    return {
        "available": False,
        "date": target_date_str,
        "formatted_date": formatted_date_str,
        "location": LOCATION,
        "temperature": None,
        "risk": None,
        "prediction": None,
        "source": source,
        "explanation": message,
        "error": message,
        "weather_source": source,
        "date_type": date_type,
        "prediction_source": "none",
    }


def _resolve_temperature_for_prediction(target: date, weather: Dict[str, Any], full_row: Dict[str, Any], is_climatological_future: bool) -> float:
    """Estimate a daily temperature using the same trained data sources as the model pipeline.
    For forecast dates, use the actual forecasted daily max temperature. For long-range future dates,
    use the climatological historical estimate derived from the same daily dataset and month/day pattern.
    The model itself is a classifier; temperature is therefore estimated from available weather data rather than
    treated as a direct model output.
    """
    if is_climatological_future:
        return float(full_row.get("max_temperature", full_row.get("mean_temperature", 30.0)))
    if weather.get("target_weather", {}).get("max_temperature") is not None:
        return float(weather["target_weather"]["max_temperature"])
    if full_row.get("max_temperature") is not None:
        return float(full_row["max_temperature"])
    if weather.get("current", {}).get("temperature_2m") is not None:
        return float(weather["current"]["temperature_2m"])
    return float(full_row.get("mean_temperature", 30.0))


def predict_heatwave(requested_date: str | date) -> Dict[str, Any]:
    """
    Main heatwave prediction pipeline for a user-provided date.
    Supports short-term Open-Meteo forecasts and long-term climatological ML predictions.
    """
    parsed = parse_date_input(requested_date)
    if not parsed["valid"]:
        today = get_current_today()
        return _failure_result(
            today.isoformat(),
            format_display_date(today),
            f"Invalid date: {parsed['error']}",
            "Date Parser",
        )

    target = parsed["target_date"]
    date_type = parsed["date_type"]  # TODAY, FUTURE, PAST
    formatted_date = parsed["formatted_date"]
    iso_date = parsed["iso_date"]

    is_climatological_future = False

    # 1. Fetch weather forecast or historical data
    weather = fetch_weather_for_date(target)

    # Check forecast range
    if not weather.get("available"):
        if date_type == "PAST":
            if not HISTORICAL_DAILY_PATH.exists():
                return _failure_result(iso_date, formatted_date, f"Historical weather data for {formatted_date} is not available.", "Historical Dataset", date_type="past")
            hist_df = pd.read_csv(HISTORICAL_DAILY_PATH, parse_dates=["date"])
            target_ts = pd.Timestamp(target).normalize()
            hist_match = hist_df[hist_df["date"] == target_ts]
            if hist_match.empty:
                return _failure_result(iso_date, formatted_date, f"Historical weather data for {formatted_date} is not available.", "Historical Dataset", date_type="past")
            daily_df = hist_df
        else:
            # LONG-TERM FUTURE DATE (>16 DAYS BEYOND OPEN-METEO WINDOW): Use climatological feature engineering
            is_climatological_future = True
    else:
        daily_df = weather["daily_df"]

    # 2. Prepare Model Features (35 features matching training)
    try:
        if is_climatological_future:
            feature_row, full_row = build_climatological_features(target)
        else:
            feature_row, full_row = build_model_features(target, daily_df)
    except Exception as exc:
        return _failure_result(iso_date, formatted_date, f"Feature preparation failed: {exc}", "Feature Engineering", date_type=date_type.lower())

    # 3. Load Trained Heatwave Model
    if not MODEL_PATH.exists():
        return _failure_result(iso_date, formatted_date, f"Trained ML model is missing at {MODEL_PATH}", "ML Model", date_type=date_type.lower())

    try:
        model = joblib.load(MODEL_PATH)
        pred_class = int(model.predict(feature_row)[0])
        probability = float(model.predict_proba(feature_row)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
        risk = calculate_risk_level(probability)

        # Standardize Moderate / Medium naming for UI display
        display_risk = "Moderate" if risk.upper() in {"MEDIUM", "MODERATE"} else risk.title()

        # Extract Temperature. The trained heatwave model is a classifier, so the daily temperature is derived
        # from the same reliable surface weather data/historical climatology used to build the feature vector.
        if is_climatological_future:
            temperature = _resolve_temperature_for_prediction(target, weather, full_row, True)
            prediction_source = "historical_model"
            source_name = "Historical dataset + trained ML model"
            weather_source = "Erode Climatological Dataset (2000-2026)"
        elif "target_weather" in weather and "max_temperature" in weather["target_weather"]:
            temperature = _resolve_temperature_for_prediction(target, weather, full_row, False)
            prediction_source = "forecast"
            source_name = "Open-Meteo forecast + trained ML model"
            weather_source = weather.get("weather_source", "Open-Meteo forecast API")
        elif "max_temperature" in full_row:
            temperature = _resolve_temperature_for_prediction(target, weather, full_row, False)
            prediction_source = "forecast" if date_type == "FUTURE" else "historical"
            source_name = "Open-Meteo forecast + trained ML model" if date_type == "FUTURE" else "Historical weather + trained ML model"
            weather_source = weather.get("weather_source", "Open-Meteo API")
        elif date_type == "TODAY" and weather.get("current", {}).get("temperature_2m") is not None:
            temperature = float(weather["current"]["temperature_2m"])
            prediction_source = "forecast"
            source_name = "Open-Meteo current weather + trained ML model"
            weather_source = weather.get("weather_source", "Open-Meteo current weather")
        else:
            temperature = _resolve_temperature_for_prediction(target, weather, full_row, is_climatological_future)
            prediction_source = "forecast" if date_type == "FUTURE" else "historical"
            source_name = "Open-Meteo forecast + trained ML model" if date_type == "FUTURE" else "Historical weather + trained ML model"
            weather_source = weather.get("weather_source", "Open-Meteo forecast API")

        result = {
            "available": True,
            "date": iso_date,
            "formatted_date": formatted_date,
            "location": LOCATION,
            "temperature": round(temperature, 1),
            "risk": display_risk,
            "probability": round(probability, 4),
            "prediction_class": pred_class,
            "prediction": "Heatwave likely" if pred_class == 1 else "Heatwave unlikely",
            "source": source_name,
            "weather_source": weather_source,
            "prediction_source": prediction_source,
            "date_type": date_type.lower(),
            "model_name": type(model).__name__,
            "forecast_range": weather.get("forecast_range"),
        }

        # Format exact user response text
        result["user_response"] = format_user_response(result)

        # Log debug info if enabled
        _log_debug_info(parsed, weather, feature_row, pred_class, display_risk)

        return result

        # Format exact user response text
        result["user_response"] = format_user_response(result)

        # Log debug info if enabled
        _log_debug_info(parsed, weather, feature_row, pred_class, display_risk)

        return result

    except Exception as exc:
        return _failure_result(iso_date, formatted_date, f"Model inference failed: {exc}", "ML Inference", date_type=date_type.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Erode heatwave risk for a requested date.")
    parser.add_argument("date", help="Target date e.g. 'today', 'tomorrow', '2026-09-20', 'September 20, 2026', '20/09/2026'")
    args = parser.parse_args()

    # Enable debug output for direct CLI invocation
    os.environ["SAFEGRAPH_AI_DEBUG"] = "1"
    res = predict_heatwave(args.date)
    print(res.get("user_response", res))


if __name__ == "__main__":
    main()