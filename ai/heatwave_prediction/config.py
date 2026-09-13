from __future__ import annotations

from pathlib import Path

from utils.constants import ERODE_LAT, ERODE_LON, ERODE_TIMEZONE, OPEN_METEO_FORECAST_URL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "heatwave_model.pkl"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
HISTORICAL_DAILY_PATH = PROJECT_ROOT / "data" / "processed" / "erode_daily.csv"

LOCATION = "Erode"
LATITUDE = ERODE_LAT
LONGITUDE = ERODE_LON
TIMEZONE = ERODE_TIMEZONE
OPEN_METEO_URL = OPEN_METEO_FORECAST_URL
MAX_FORECAST_DAYS = 16

REQUESTED_HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "vapour_pressure_deficit",
    "cloud_cover",
    "dew_point_2m",
)