from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import load_pickle, resolve_path  # noqa: E402
from features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features  # noqa: E402


MODEL_PATH = resolve_path("models", "heatwave_model.pkl")
FEATURE_COLUMNS_PATH = resolve_path("models", "feature_columns.pkl")
TRAINING_META_PATH = resolve_path("models", "training_metadata.pkl")


DEFAULT_DAILY_CSV = resolve_path("data", "processed", "erode_daily.csv")


def load_history_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}. Run preprocessing first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def build_prediction_features(history_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "date",
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
    missing = [column for column in required_columns if column not in history_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in prediction input: {missing}")

    feature_df = history_df.copy().sort_values("date").reset_index(drop=True)
    feature_df = add_calendar_features(feature_df)
    feature_df = add_lag_features(feature_df)
    feature_df = add_rolling_features(feature_df)
    return feature_df


def calculate_risk_level(probability: float) -> str:
    if probability >= 0.80:
        return "HIGH"
    if probability >= 0.50:
        return "MEDIUM"
    return "LOW"


def predict_latest(history_df: pd.DataFrame) -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Run python training/train.py first.")

    model = joblib.load(MODEL_PATH)
    feature_columns = load_pickle(FEATURE_COLUMNS_PATH)
    feature_df = build_prediction_features(history_df)
    feature_row = feature_df.tail(1).copy()
    feature_row = feature_row.dropna(subset=feature_columns)
    if feature_row.empty:
        raise ValueError(
            "Not enough history to build the lag and rolling features. Provide at least 8 days of daily data."
        )

    x_input = feature_row[feature_columns]
    prediction = int(model.predict(x_input)[0])
    probability = float(model.predict_proba(x_input)[0][1]) if hasattr(model, "predict_proba") else float("nan")
    metadata = load_pickle(TRAINING_META_PATH) if TRAINING_META_PATH.exists() else {}
    risk_level = calculate_risk_level(probability if np.isfinite(probability) else 0.0)

    target_date = str(pd.Timestamp(feature_row["date"].values[0]).date())
    return {
        "location": "Erode, Tamil Nadu",
        "target_date": target_date,
        "max_temperature_c": float(feature_row["max_temperature"].values[0]),
        "heatwave_prediction": prediction,
        "heatwave_probability": round(probability, 4) if np.isfinite(probability) else None,
        "risk_level": risk_level,
        "model": metadata.get("best_model"),
    }


def create_today_row(
    history_df: pd.DataFrame,
    max_temp: float,
    mean_temp: float | None = None,
    min_temp: float | None = None,
    humidity: float | None = None,
    date_str: str | None = None,
) -> pd.DataFrame:
    last_row = history_df.iloc[-1].copy()
    next_date = pd.Timestamp.now().floor("D") if date_str is None else pd.Timestamp(date_str)
    if next_date <= last_row["date"]:
        next_date = last_row["date"] + pd.Timedelta(days=1)

    mean_temp_val = mean_temp if mean_temp is not None else round(max_temp - 5.5, 1)
    min_temp_val = min_temp if min_temp is not None else round(max_temp - 10.5, 1)
    humidity_val = humidity if humidity is not None else float(last_row.get("mean_relative_humidity", 60.0))

    new_row = {
        "date": next_date,
        "mean_temperature": mean_temp_val,
        "max_temperature": max_temp,
        "min_temperature": min_temp_val,
        "mean_relative_humidity": humidity_val,
        "max_relative_humidity": min(100.0, humidity_val + 20.0),
        "mean_wind_speed": float(last_row.get("mean_wind_speed", 8.0)),
        "max_wind_speed": float(last_row.get("max_wind_speed", 14.0)),
        "max_wind_gust": float(last_row.get("max_wind_gust", 24.0)),
        "precipitation_sum": 0.0,
        "mean_vpd": float(last_row.get("mean_vpd", 2.0)),
        "max_vpd": float(last_row.get("max_vpd", 3.0)),
        "mean_cloud_cover": float(last_row.get("mean_cloud_cover", 30.0)),
        "mean_dew_point": float(last_row.get("mean_dew_point", 16.0)),
        "mean_apparent_temperature": mean_temp_val + 1.5,
        "max_apparent_temperature": max_temp + 2.0,
        "min_apparent_temperature": min_temp_val + 1.0,
    }

    updated_df = pd.concat([history_df, pd.DataFrame([new_row])], ignore_index=True)
    return updated_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an Erode heatwave from recent daily weather history.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_DAILY_CSV),
        help=f"Path to a daily weather history CSV (default: {DEFAULT_DAILY_CSV}).",
    )
    parser.add_argument("--max-temp", type=float, help="Today's maximum temperature in °C.")
    parser.add_argument("--mean-temp", type=float, help="Today's average temperature in °C (optional).")
    parser.add_argument("--min-temp", type=float, help="Today's minimum temperature in °C (optional).")
    parser.add_argument("--humidity", type=float, help="Today's relative humidity in % (optional).")
    parser.add_argument("--date", type=str, help="Target prediction date (YYYY-MM-DD, optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.input_csv)
    history_df = load_history_dataframe(csv_path)

    if args.max_temp is not None:
        history_df = create_today_row(
            history_df,
            max_temp=args.max_temp,
            mean_temp=args.mean_temp,
            min_temp=args.min_temp,
            humidity=args.humidity,
            date_str=args.date,
        )

    result = predict_latest(history_df)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

