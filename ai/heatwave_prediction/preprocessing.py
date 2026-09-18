from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Tuple

import pandas as pd

from common import load_pickle
from features.feature_engineering import add_calendar_features, add_lag_features, add_rolling_features

from .config import FEATURE_COLUMNS_PATH, HISTORICAL_DAILY_PATH


def load_feature_columns() -> List[str]:
    """Load saved feature column list matching model training."""
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature metadata file missing at: {FEATURE_COLUMNS_PATH}")
    columns = load_pickle(FEATURE_COLUMNS_PATH)
    if not isinstance(columns, list) or not columns:
        raise ValueError("Saved feature metadata is empty or invalid.")
    return columns


def build_model_features(target_date: date, forecast_daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Engineer the exact 35 model features for a target date without data leakage.
    Stitches historical observations with forecast daily rows to generate accurate
    lags (1, 2, 3, 7) and rolling averages (3, 5, 7).
    """
    if not HISTORICAL_DAILY_PATH.exists():
        raise FileNotFoundError(f"Historical daily dataset missing at: {HISTORICAL_DAILY_PATH}")

    history = pd.read_csv(HISTORICAL_DAILY_PATH, parse_dates=["date"]).sort_values("date")
    target_ts = pd.Timestamp(target_date).normalize()

    # Filter history strictly prior to target date
    history_before_target = history[history["date"] < target_ts].tail(60)

    if history_before_target.empty:
        # If target date is within historical dataset itself (past date query)
        target_in_history = history[history["date"] == target_ts]
        if not target_in_history.empty:
            history_before_target = history[history["date"] <= target_ts].tail(60)
            combined = history_before_target.copy().sort_values("date").reset_index(drop=True)
        else:
            raise ValueError(f"Not enough historical daily records prior to {target_date.isoformat()} for lag calculation.")
    else:
        history_end = history_before_target["date"].max()
        future_rows = forecast_daily[forecast_daily["date"] > history_end].copy()
        combined = pd.concat([history_before_target, future_rows], ignore_index=True).sort_values("date").reset_index(drop=True)

    # Compute calendar, lag, and rolling features
    combined = add_calendar_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)

    # Extract target date row
    target_rows = combined[combined["date"].dt.date == target_date]
    if target_rows.empty:
        raise ValueError(f"Feature calculation failed: no row generated for target date {target_date.isoformat()}.")

    feature_columns = load_feature_columns()
    missing_cols = [col for col in feature_columns if col not in target_rows.columns]
    if missing_cols:
        raise ValueError(f"Missing required model feature columns: {missing_cols}")

    feature_row = target_rows[feature_columns].copy()
    if feature_row.isna().any().any():
        null_cols = feature_row.columns[feature_row.isna().any()].tolist()
        raise ValueError(f"Required model features contain NaN values for {target_date.isoformat()}: {null_cols}")

    return feature_row, target_rows.iloc[0].to_dict()


def build_climatological_features(target_date: date) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Engineer model features for a long-term future date (beyond 16 days forecast)
    using climatological historical averages from erode_daily.csv.
    """
    if not HISTORICAL_DAILY_PATH.exists():
        raise FileNotFoundError(f"Historical daily dataset missing at: {HISTORICAL_DAILY_PATH}")

    history = pd.read_csv(HISTORICAL_DAILY_PATH, parse_dates=["date"]).sort_values("date")
    history["month_day"] = history["date"].dt.strftime("%m-%d")

    # Generate a sequence of 30 daily dates leading up to and including target_date
    start_date = pd.Timestamp(target_date) - pd.Timedelta(days=30)
    end_date = pd.Timestamp(target_date)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    base_cols = [
        "mean_temperature", "max_temperature", "min_temperature",
        "mean_relative_humidity", "max_relative_humidity",
        "mean_wind_speed", "max_wind_speed", "max_wind_gust",
        "precipitation_sum", "mean_vpd", "max_vpd",
        "mean_cloud_cover", "mean_dew_point",
        "mean_apparent_temperature", "max_apparent_temperature", "min_apparent_temperature"
    ]

    # Pre-calculate climatological mean by month_day across all historical years
    clim_means = history.groupby("month_day")[base_cols].mean()
    overall_means = history[base_cols].mean()

    rows = []
    for d in date_range:
        md = d.strftime("%m-%d")
        if md in clim_means.index:
            vals = clim_means.loc[md].to_dict()
        else:
            vals = overall_means.to_dict()
        vals["date"] = d
        rows.append(vals)

    seq_df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Compute calendar, lag, and rolling features
    seq_df = add_calendar_features(seq_df)
    seq_df = add_lag_features(seq_df)
    seq_df = add_rolling_features(seq_df)

    target_rows = seq_df[seq_df["date"].dt.date == target_date]
    if target_rows.empty:
        raise ValueError(f"Climatological feature calculation failed for target date {target_date.isoformat()}.")

    feature_columns = load_feature_columns()
    feature_row = target_rows[feature_columns].copy()

    # Fill any NaNs in initial window rows if any
    if feature_row.isna().any().any():
        feature_row = feature_row.bfill().ffill().fillna(0)

    return feature_row, target_rows.iloc[0].to_dict()