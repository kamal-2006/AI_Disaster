from __future__ import annotations

import pandas as pd
import streamlit as st
from common import resolve_path

DAILY_CSV = resolve_path("data", "processed", "erode_daily.csv")
FEATURES_CSV = resolve_path("data", "processed", "erode_features.csv")


@st.cache_data(ttl=3600)
def load_historical_daily_data() -> pd.DataFrame:
    """Load preprocessed daily weather dataset for Erode (2000–present)."""
    if not DAILY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DAILY_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600)
def load_historical_features_data() -> pd.DataFrame:
    """Load feature engineering dataset containing labeled heatwave events."""
    if not FEATURES_CSV.exists():
        # Fallback to daily data
        return load_historical_daily_data()
    df = pd.read_csv(FEATURES_CSV, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def get_yearly_heatwave_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly heatwave event count and average max temperature."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["year"] = df["date"].dt.year
    hw_col = "heatwave" if "heatwave" in df.columns else "heatwave_event" if "heatwave_event" in df.columns else None

    if hw_col:
        yearly = df.groupby("year").agg(
            total_days=("date", "count"),
            heatwave_days=(hw_col, "sum"),
            avg_max_temp=("max_temperature", "mean"),
            peak_temp=("max_temperature", "max"),
        )
        yearly["heatwave_rate_pct"] = (yearly["heatwave_days"] / yearly["total_days"] * 100).round(2)
    else:
        # If heatwave label not present, derive based on 95th percentile (>38.5 C)
        df["is_hot"] = (df["max_temperature"] >= 38.5).astype(int)
        yearly = df.groupby("year").agg(
            total_days=("date", "count"),
            heatwave_days=("is_hot", "sum"),
            avg_max_temp=("max_temperature", "mean"),
            peak_temp=("max_temperature", "max"),
        )
        yearly["heatwave_rate_pct"] = (yearly["heatwave_days"] / yearly["total_days"] * 100).round(2)

    return yearly.reset_index()


def get_monthly_heatwave_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly heatwave distribution across all historical years."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    hw_col = "heatwave" if "heatwave" in df.columns else "heatwave_event" if "heatwave_event" in df.columns else None

    if not hw_col:
        df["is_hot"] = (df["max_temperature"] >= 38.5).astype(int)
        hw_col = "is_hot"

    monthly = df.groupby(["month", "month_name"]).agg(
        total_days=("date", "count"),
        heatwave_days=(hw_col, "sum"),
        avg_max_temp=("max_temperature", "mean"),
        max_temp_recorded=("max_temperature", "max"),
        avg_humidity=("mean_relative_humidity", "mean"),
    ).reset_index().sort_values("month")

    monthly["heatwave_rate_pct"] = (monthly["heatwave_days"] / monthly["total_days"] * 100).round(2)
    return monthly


def get_heatwave_calendar_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Generate Year x Month heatwave count pivot table for heatmaps."""
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")
    hw_col = "heatwave" if "heatwave" in df.columns else "heatwave_event" if "heatwave_event" in df.columns else None

    if not hw_col:
        df["is_hot"] = (df["max_temperature"] >= 38.5).astype(int)
        hw_col = "is_hot"

    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = df.pivot_table(index="year", columns="month", values=hw_col, aggfunc="sum", fill_value=0)
    existing_months = [m for m in months_order if m in pivot.columns]
    pivot = pivot[existing_months]
    return pivot
