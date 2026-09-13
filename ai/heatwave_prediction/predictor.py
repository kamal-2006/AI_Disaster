from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import joblib

from services.prediction_service import calculate_risk_level, get_realtime_prediction

from .config import HISTORICAL_DAILY_PATH, LOCATION, MODEL_PATH, TIMEZONE
from .preprocessing import build_model_features
from .weather_service import fetch_weather_for_date


MONTHS = {name.lower(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
    start=1,
)}


def _today() -> date:
    return datetime.now(ZoneInfo(TIMEZONE)).date()


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    text = value.strip()
    today = _today()
    lowered = text.lower()
    if lowered in {"today", "now", "current"}:
        return today
    if lowered == "tomorrow":
        return today + timedelta(days=1)
    if lowered == "yesterday":
        return today - timedelta(days=1)

    for format_string in ("%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%d %B %Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, format_string).date()
        except ValueError:
            continue
    raise ValueError("Use YYYY-MM-DD, Today, Tomorrow, September 20, 2026, or 20/09/2026.")


def _failure(target: date, message: str, source: str = "Heatwave prediction module") -> dict[str, Any]:
    return {
        "available": False,
        "date": target.isoformat(),
        "location": LOCATION,
        "temperature": None,
        "risk": None,
        "prediction": None,
        "explanation": message,
        "weather_source": source,
    }


def _result(target: date, temperature: float, risk: str, probability: float, prediction: str, source: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": True,
        "date": target.isoformat(),
        "location": LOCATION,
        "temperature": round(float(temperature), 1),
        "risk": risk,
        "probability": round(float(probability), 4),
        "prediction": prediction,
        "explanation": f"The {'forecast' if target > _today() else 'observed'} weather conditions indicate a {risk.lower()} heatwave risk from the trained model.",
        "weather_source": source,
        **extra,
    }


def predict_heatwave(requested_date: str | date) -> dict[str, Any]:
    """Predict heatwave risk for a user-provided date using the existing model."""
    try:
        target = parse_date(requested_date)
    except (TypeError, ValueError) as exc:
        return _failure(_today(), f"Invalid date: {exc}")

    today = _today()
    if target == today:
        prediction = get_realtime_prediction()
        if prediction.get("error"):
            return _failure(target, f"Model prediction failed: {prediction['error']}", prediction.get("weather_source", "Current weather API"))
        current = prediction.get("current_weather", {})
        temperature = current.get("temperature_2m")
        if temperature is None or prediction.get("risk_level") is None:
            return _failure(target, "Current weather or model risk data is incomplete.")
        probability = float(prediction.get("probability", 0.0))
        risk = str(prediction["risk_level"])
        return _result(
            target,
            float(temperature),
            risk,
            probability,
            "Heatwave likely" if risk.upper() == "HIGH" else "Heatwave risk assessed by model",
            "Open-Meteo current weather API",
            date_type="today",
            model_name=prediction.get("model_name"),
            model_features=prediction.get("feature_row"),
        )

    weather = fetch_weather_for_date(target)
    if not weather.get("available"):
        return _failure(target, weather.get("error", "Weather forecast is unavailable."), weather.get("weather_source", "Open-Meteo forecast API"))

    if target < today:
        if not HISTORICAL_DAILY_PATH.exists():
            return _failure(target, "Historical weather data is not available for this date.", "Historical Erode daily dataset")
        source = "Historical Erode daily dataset"
    else:
        source = weather.get("weather_source", "Open-Meteo forecast API")

    try:
        feature_row, full_row = build_model_features(target, weather["daily_df"])
        if not MODEL_PATH.exists():
            return _failure(target, f"Trained model is missing: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        prediction_class = int(model.predict(feature_row)[0])
        probability = float(model.predict_proba(feature_row)[0][1]) if hasattr(model, "predict_proba") else float(prediction_class)
        risk = calculate_risk_level(probability)
        temperature = float(weather["target_weather"]["max_temperature"])
        return _result(
            target,
            temperature,
            risk,
            probability,
            "Heatwave likely" if prediction_class == 1 else "Heatwave not indicated by the model",
            source,
            date_type="future" if target > today else "past",
            model_name=type(model).__name__,
            model_features=feature_row,
            engineered_row=full_row,
            forecast_range=weather.get("forecast_range"),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _failure(target, f"Model prediction failed: {exc}", source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict Erode heatwave risk for a date.")
    parser.add_argument("date", help="Today, Tomorrow, YYYY-MM-DD, September 20, 2026, or 20/09/2026")
    args = parser.parse_args()
    result = predict_heatwave(args.date)
    print(result)


if __name__ == "__main__":
    main()