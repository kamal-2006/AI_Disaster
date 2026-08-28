from __future__ import annotations

from datetime import datetime
import pandas as pd
import requests
import streamlit as st
from utils.constants import ERODE_LAT, ERODE_LON, ERODE_TIMEZONE, OPEN_METEO_FORECAST_URL


@st.cache_data(ttl=300, show_spinner=False)
def fetch_erode_weather() -> dict[str, object]:
    """
    Fetch live and past 7-day weather data for Erode, Tamil Nadu from Open-Meteo API.
    Uses st.cache_data with a 5-minute (300s) TTL.
    """
    params = {
        "latitude": ERODE_LAT,
        "longitude": ERODE_LON,
        "timezone": ERODE_TIMEZONE,
        "past_days": 7,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "vapour_pressure_deficit",
            "cloud_cover",
            "dew_point_2m",
        ],
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "vapour_pressure_deficit",
            "cloud_cover",
            "dew_point_2m",
        ],
    }

    headers = {"User-Agent": "SafeGraph-AI/1.0 (Erode Heatwave Prediction)"}

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})

        # Build hourly dataframe
        hourly_df = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly.get("time", [])),
                "temperature_2m": hourly.get("temperature_2m", []),
                "relative_humidity_2m": hourly.get("relative_humidity_2m", []),
                "apparent_temperature": hourly.get("apparent_temperature", []),
                "precipitation": hourly.get("precipitation", []),
                "wind_speed_10m": hourly.get("wind_speed_10m", []),
                "wind_gusts_10m": hourly.get("wind_gusts_10m", []),
                "vapour_pressure_deficit": hourly.get("vapour_pressure_deficit", []),
                "cloud_cover": hourly.get("cloud_cover", []),
                "dew_point_2m": hourly.get("dew_point_2m", []),
            }
        )

        current_metrics = {
            "temperature_2m": float(current.get("temperature_2m", 32.5)),
            "relative_humidity_2m": float(current.get("relative_humidity_2m", 60.0)),
            "apparent_temperature": float(current.get("apparent_temperature", 35.0)),
            "precipitation": float(current.get("precipitation", 0.0)),
            "wind_speed_10m": float(current.get("wind_speed_10m", 12.0)),
            "wind_gusts_10m": float(current.get("wind_gusts_10m", 20.0)),
            "vapour_pressure_deficit": float(current.get("vapour_pressure_deficit", 2.1)),
            "cloud_cover": float(current.get("cloud_cover", 25.0)),
            "dew_point_2m": float(current.get("dew_point_2m", 22.0)),
            "last_updated": current.get("time", datetime.now().strftime("%Y-%m-%d %H:%M")),
            "status": "Live API",
        }

        return {
            "current": current_metrics,
            "hourly_df": hourly_df,
            "error": None,
        }

    except Exception as exc:
        # Fallback dictionary if API call fails
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        return {
            "current": {
                "temperature_2m": 35.2,
                "relative_humidity_2m": 58.0,
                "apparent_temperature": 38.5,
                "precipitation": 0.0,
                "wind_speed_10m": 11.5,
                "wind_gusts_10m": 18.0,
                "vapour_pressure_deficit": 2.45,
                "cloud_cover": 20.0,
                "dew_point_2m": 21.0,
                "last_updated": now_str,
                "status": "Offline / Fallback",
            },
            "hourly_df": pd.DataFrame(),
            "error": str(exc),
        }


def convert_hourly_to_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data into daily format matching the training features.
    """
    if hourly_df.empty or "time" not in hourly_df.columns:
        return pd.DataFrame()

    df = hourly_df.copy()
    df["date"] = df["time"].dt.floor("D")

    aggregations = {
        "temperature_2m": ["mean", "max", "min"],
        "relative_humidity_2m": ["mean", "max"],
        "wind_speed_10m": ["mean", "max"],
        "wind_gusts_10m": ["max"],
        "precipitation": ["sum"],
        "vapour_pressure_deficit": ["mean", "max"],
        "cloud_cover": ["mean"],
        "dew_point_2m": ["mean"],
        "apparent_temperature": ["mean", "max", "min"],
    }

    grouped = df.groupby("date").agg(aggregations)
    grouped.columns = [
        "mean_temperature",
        "max_temperature",
        "min_temperature",
        "mean_relative_humidity",
        "max_relative_humidity",
        "mean_wind_speed",
        "max_wind_speed",
        "max_wind_gust",
        "precipitation_sum",
        "mean_vpd",
        "max_vpd",
        "mean_cloud_cover",
        "mean_dew_point",
        "mean_apparent_temperature",
        "max_apparent_temperature",
        "min_apparent_temperature",
    ]
    return grouped.reset_index()
