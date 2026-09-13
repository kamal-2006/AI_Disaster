from __future__ import annotations

from datetime import date

import pandas as pd

from common import load_pickle
from features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features

from .config import FEATURE_COLUMNS_PATH, HISTORICAL_DAILY_PATH


def load_feature_columns() -> list[str]:
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature metadata is missing: {FEATURE_COLUMNS_PATH}")
    columns = load_pickle(FEATURE_COLUMNS_PATH)
    if not isinstance(columns, list) or not columns:
        raise ValueError("Saved feature metadata is empty or invalid.")
    return columns


def build_model_features(target_date: date, forecast_daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply the exact training feature functions and saved feature order."""
    if not HISTORICAL_DAILY_PATH.exists():
        raise FileNotFoundError(f"Historical daily dataset is missing: {HISTORICAL_DAILY_PATH}")

    history = pd.read_csv(HISTORICAL_DAILY_PATH, parse_dates=["date"]).sort_values("date")
    target = pd.Timestamp(target_date).normalize()
    history_before_target = history[history["date"] < target].tail(60)
    if history_before_target.empty:
        raise ValueError("There is not enough historical data before the requested date to calculate lag features.")

    history_end = history_before_target["date"].max()
    future_rows = forecast_daily[forecast_daily["date"] > history_end].copy()
    combined = pd.concat([history_before_target, future_rows], ignore_index=True).sort_values("date")
    combined = add_calendar_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    target_rows = combined[combined["date"] == target]
    if target_rows.empty:
        raise ValueError(f"No model feature row was produced for {target_date.isoformat()}.")

    feature_columns = load_feature_columns()
    missing = [column for column in feature_columns if column not in target_rows.columns]
    if missing:
        raise ValueError(f"The model requires missing features: {missing}")
    feature_row = target_rows[feature_columns].copy()
    if feature_row.isna().any().any():
        missing_values = feature_row.columns[feature_row.isna().any()].tolist()
        raise ValueError(f"Required model features contain missing values: {missing_values}")
    return feature_row, target_rows.iloc[0].to_dict()