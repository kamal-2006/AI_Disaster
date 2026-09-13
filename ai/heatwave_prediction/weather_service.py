from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import requests

from services.weather_api import convert_hourly_to_daily_summary

from .config import (
    LATITUDE,
    LOCATION,
    LONGITUDE,
    MAX_FORECAST_DAYS,
    OPEN_METEO_URL,
    REQUESTED_HOURLY_VARIABLES,
    TIMEZONE,
)


def _hourly_frame(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    columns = {"time": pd.to_datetime(hourly.get("time", []))}
    for variable in REQUESTED_HOURLY_VARIABLES:
        columns[variable] = hourly.get(variable, [])
    frame = pd.DataFrame(columns)
    if not frame.empty:
        frame = frame.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return frame


def fetch_weather_for_date(target_date: date) -> dict[str, Any]:
    """Fetch Open-Meteo data and the daily model row for one requested date."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "past_days": 7,
        "forecast_days": MAX_FORECAST_DAYS,
        "current": list(REQUESTED_HOURLY_VARIABLES),
        "hourly": list(REQUESTED_HOURLY_VARIABLES),
    }
    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            headers={"User-Agent": "SafeGraph-AI-Heatwave-Prediction/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        hourly_df = _hourly_frame(payload)
        daily_df = convert_hourly_to_daily_summary(hourly_df)
        target_row = daily_df[daily_df["date"].dt.date == target_date].copy()
        current = payload.get("current", {})
        if target_row.empty:
            return {
                "available": False,
                "error": "Weather forecast is not currently available for this date, so a reliable heatwave prediction cannot be generated.",
                "location": LOCATION,
                "target_date": target_date.isoformat(),
                "forecast_range": {
                    "start": daily_df["date"].min().date().isoformat() if not daily_df.empty else None,
                    "end": daily_df["date"].max().date().isoformat() if not daily_df.empty else None,
                },
                "weather_source": "Open-Meteo forecast API",
            }

        return {
            "available": True,
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "current": current,
            "hourly_df": hourly_df,
            "daily_df": daily_df,
            "target_weather": target_row.iloc[0].to_dict(),
            "forecast_range": {
                "start": daily_df["date"].min().date().isoformat() if not daily_df.empty else None,
                "end": daily_df["date"].max().date().isoformat() if not daily_df.empty else None,
            },
            "weather_source": "Open-Meteo forecast API",
        }
    except requests.RequestException as exc:
        return {
            "available": False,
            "error": f"Open-Meteo request failed: {exc}",
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "weather_source": "Open-Meteo forecast API",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "error": f"Open-Meteo returned unusable weather data: {exc}",
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "weather_source": "Open-Meteo forecast API",
        }