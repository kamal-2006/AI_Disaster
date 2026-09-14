from __future__ import annotations

"""Date-aware routing between current weather, forecast, and historical data."""

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from services.prediction_service import get_prediction_for_date, get_realtime_prediction
from utils.constants import ERODE_TIMEZONE, LOCATION_NAME


MONTHS = {name.lower(): index for index, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
    start=1,
)}
MONTH_PATTERN = "|".join(MONTHS)
DATE_PATTERN = re.compile(
    rf"\b(?:(?P<month>{MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:,\s*|\s+)(?P<year>\d{{4}})|"
    rf"(?P<day_first>\d{{1,2}})\s+(?P<month_first>{MONTH_PATTERN})(?:\s+(?P<year_first>\d{{4}}))?|"
    rf"(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})|(?P<numeric>\d{{1,2}}/\d{{1,2}}/\d{{4}}))\b",
    flags=re.IGNORECASE,
)


from heatwave_prediction.date_parser import parse_date_input


def erode_today() -> date:
    return datetime.now(ZoneInfo(ERODE_TIMEZONE)).date()


def parse_requested_date(query: str, today: Optional[date] = None) -> Dict[str, Any]:
    """Parse explicit and relative dates using heatwave_prediction date_parser."""
    res = parse_date_input(query, today_override=today)
    if res["valid"]:
        return {
            "date": res["target_date"],
            "date_type": res["date_type"].lower(),
            "raw": res["raw_input"],
            "error": None,
        }
    return {"date": None, "date_type": "invalid", "raw": query, "error": res["error"]}


def classify_weather_question(query: str) -> Optional[Dict[str, Any]]:
    """Return weather routing metadata, or None for Knowledge Graph-only questions."""
    q_lower = query.lower()
    weather_terms = (
        "temperature", "weather", "forecast", "heatwave risk", "heat wave risk", "heatwave prediction",
        "will there be a heatwave", "current heatwave", "heatwave today", "heatwave tomorrow",
    )
    heatwave_request = "heatwave" in q_lower or "heat wave" in q_lower
    date_words = any(term in q_lower for term in ("today", "tomorrow", "yesterday", "on "))
    heatwave_date_request = heatwave_request and any(term in q_lower for term in ("risk", "prediction", "will there", "is there"))

    parsed_res = parse_date_input(query)
    is_explicit_date = parsed_res["valid"]

    if not any(term in q_lower for term in weather_terms) and not (heatwave_date_request or (heatwave_request and date_words)) and not is_explicit_date:
        return None

    parsed_date = parse_requested_date(query)
    date_type = parsed_date["date_type"]
    if date_type == "invalid":
        intent = "invalid_date"
    elif date_type == "future":
        intent = "future_heatwave_risk"
    elif date_type == "past":
        intent = "historical_heatwave_risk"
    else:
        intent = "current_heatwave_risk"

    return {
        "location": LOCATION_NAME,
        "date": parsed_date["date"].isoformat() if parsed_date["date"] else None,
        "date_type": date_type,
        "date_raw": parsed_date["raw"],
        "date_error": parsed_date["error"],
        "intent": intent,
        "question": query,
    }


def get_weather_result(weather_request: Dict[str, Any]) -> Dict[str, Any]:
    """Retrieve authoritative weather and, when possible, run the trained model."""
    if weather_request["date_type"] == "invalid":
        return {"available": False, "error": f"The requested date is invalid: {weather_request['date_error']}.", **weather_request}

    target_date = date.fromisoformat(weather_request["date"])
    if weather_request["date_type"] == "today":
        prediction = get_realtime_prediction()
        current = prediction.get("current_weather", {})
        return {
            "available": not bool(prediction.get("error")),
            "weather_source": "Open-Meteo current weather API",
            "weather_error": prediction.get("weather_error"),
            "temperature": current.get("temperature_2m"),
            "weather": current,
            "risk_level": prediction.get("risk_level"),
            "probability": prediction.get("probability"),
            "prediction": prediction,
            **weather_request,
        }

    prediction = get_prediction_for_date(target_date)
    return {
        **prediction,
        **weather_request,
        "temperature": prediction.get("temperature"),
        "risk_level": prediction.get("risk_level"),
        "probability": prediction.get("probability"),
    }