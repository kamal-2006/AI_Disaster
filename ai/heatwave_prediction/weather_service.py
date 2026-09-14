from __future__ import annotations

from datetime import date
from typing import Any, Dict

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
from .date_parser import format_display_date, get_current_today


def fetch_weather_for_date(target_date: date) -> Dict[str, Any]:
    """
    Fetch Open-Meteo forecast or historical weather for a requested calendar date.
    Calculates forecast daily rows dynamically for model feature engineering.
    """
    today = get_current_today()
    hourly_vars_str = ",".join(REQUESTED_HOURLY_VARIABLES)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "past_days": 7,
        "forecast_days": MAX_FORECAST_DAYS,
        "current": hourly_vars_str,
        "hourly": hourly_vars_str,
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

        hourly_raw = payload.get("hourly", {})
        time_series = pd.to_datetime(hourly_raw.get("time", []))
        if len(time_series) == 0:
            raise ValueError("Open-Meteo returned empty hourly timestamps.")

        columns: dict[str, Any] = {"time": time_series}
        for variable in REQUESTED_HOURLY_VARIABLES:
            columns[variable] = hourly_raw.get(variable, [])

        hourly_df = pd.DataFrame(columns).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
        daily_df = convert_hourly_to_daily_summary(hourly_df)

        if daily_df.empty:
            raise ValueError("Failed to aggregate hourly weather into daily summaries.")

        forecast_start = daily_df["date"].min().date()
        forecast_end = daily_df["date"].max().date()

        # Check if requested date is within the forecast window
        target_row = daily_df[daily_df["date"].dt.date == target_date].copy()

        if target_row.empty:
            formatted_target = format_display_date(target_date)
            if target_date > forecast_end:
                return {
                    "available": False,
                    "error": f"Forecast data is not available for {formatted_target} yet, so a reliable date-specific heatwave prediction cannot be generated.",
                    "location": LOCATION,
                    "target_date": target_date.isoformat(),
                    "forecast_range": {
                        "start": forecast_start.isoformat(),
                        "end": forecast_end.isoformat(),
                    },
                    "weather_source": "Open-Meteo forecast API",
                    "api_url": response.url,
                }
            elif target_date < forecast_start:
                return {
                    "available": False,
                    "error": f"Date {target_date.isoformat()} is prior to available live forecast buffer.",
                    "location": LOCATION,
                    "target_date": target_date.isoformat(),
                    "is_past": True,
                    "weather_source": "Historical Erode daily dataset",
                    "api_url": response.url,
                }

        current = payload.get("current", {})
        target_weather = target_row.iloc[0].to_dict()

        return {
            "available": True,
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "current": current,
            "hourly_df": hourly_df,
            "daily_df": daily_df,
            "target_weather": target_weather,
            "forecast_range": {
                "start": forecast_start.isoformat(),
                "end": forecast_end.isoformat(),
            },
            "weather_source": "Open-Meteo forecast API",
            "api_url": response.url,
        }

    except requests.RequestException as exc:
        return {
            "available": False,
            "error": f"Open-Meteo API connection failure: {exc}",
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "weather_source": "Open-Meteo forecast API",
        }
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "available": False,
            "error": f"Weather data processing failure: {exc}",
            "location": LOCATION,
            "target_date": target_date.isoformat(),
            "weather_source": "Open-Meteo forecast API",
        }