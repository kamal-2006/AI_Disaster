from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils.constants import COLOR_ACCENT, COLOR_HIGH_RISK, COLOR_LOW_RISK, COLOR_MEDIUM_RISK

# Shared dark theme layout
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.0)",
    plot_bgcolor="rgba(30, 41, 59, 0.5)",
    font=dict(color="#F8FAFC", family="sans-serif"),
    margin=dict(l=40, r=30, t=40, b=40),
    xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
    yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
)


def plot_hourly_forecast(hourly_df: pd.DataFrame):
    """Render interactive multi-metric hourly weather trends (Temp, Humidity, VPD, Wind)."""
    if hourly_df.empty or "time" not in hourly_df.columns:
        st.info("No hourly forecast data available.")
        return

    df = hourly_df.head(48).copy()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Temperature (°C) & Apparent Temp",
            "Relative Humidity (%)",
            "Vapour Pressure Deficit (kPa)",
            "Wind Speed & Gusts (km/h)",
        ),
    )

    # Subplot 1: Temp & Apparent Temp
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["temperature_2m"], name="Temperature", line=dict(color="#F59E0B", width=2.5)),
        row=1,
        col=1,
    )
    if "apparent_temperature" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["apparent_temperature"],
                name="Apparent Temp",
                line=dict(color="#EF4444", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Relative Humidity
    fig.add_trace(
        go.Scatter(
            x=df["time"],
            y=df["relative_humidity_2m"],
            name="Humidity",
            line=dict(color="#3B82F6", width=2),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.1)",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: VPD
    if "vapour_pressure_deficit" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"],
                y=df["vapour_pressure_deficit"],
                name="VPD",
                line=dict(color="#10B981", width=2),
            ),
            row=2,
            col=1,
        )

    # Subplot 4: Wind Speed
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["wind_speed_10m"], name="Wind Speed", line=dict(color="#8B5CF6", width=2)),
        row=2,
        col=2,
    )
    if "wind_gusts_10m" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["time"], y=df["wind_gusts_10m"], name="Wind Gusts", line=dict(color="#C084FC", width=1, dash="dot")
            ),
            row=2,
            col=2,
        )

    fig.update_layout(**DARK_LAYOUT, height=450, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)


def plot_yearly_heatwave_frequency(yearly_df: pd.DataFrame):
    """Bar chart of yearly heatwave days in Erode (2000–present)."""
    if yearly_df.empty:
        st.info("No historical data available for yearly analysis.")
        return

    fig = px.bar(
        yearly_df,
        x="year",
        y="heatwave_days",
        color="avg_max_temp",
        color_continuous_scale="OrRd",
        labels={"year": "Year", "heatwave_days": "Heatwave Days", "avg_max_temp": "Avg Max Temp (°C)"},
        title="Yearly Heatwave Event Frequency (Erode, TN)",
    )
    fig.update_layout(**DARK_LAYOUT, height=380)
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_heatwave_distribution(monthly_df: pd.DataFrame):
    """Bar chart showing historical heatwave concentration by month."""
    if monthly_df.empty:
        st.info("No monthly distribution data available.")
        return

    fig = px.bar(
        monthly_df,
        x="month_name",
        y="heatwave_days",
        color="max_temp_recorded",
        color_continuous_scale="YlOrRd",
        labels={"month_name": "Month", "heatwave_days": "Total Heatwave Days", "max_temp_recorded": "Peak Temp (°C)"},
        title="Monthly Heatwave Distribution (Peak Heat Stress Months)",
    )
    fig.update_layout(**DARK_LAYOUT, height=380)
    st.plotly_chart(fig, use_container_width=True)


def plot_rolling_temperature_trend(daily_df: pd.DataFrame):
    """Multi-year max temperature trend line with 30-day rolling average."""
    if daily_df.empty or "date" not in daily_df.columns:
        st.info("No daily temperature data available.")
        return

    df = daily_df.copy()
    df["rolling_30d"] = df["max_temperature"].rolling(30, min_periods=5).mean()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["max_temperature"],
            name="Daily Max Temp",
            line=dict(color="rgba(245, 158, 11, 0.25)", width=1),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["rolling_30d"],
            name="30-Day Rolling Avg",
            line=dict(color="#EF4444", width=2.5),
        )
    )
    fig.update_layout(
        **DARK_LAYOUT,
        height=400,
        title="Erode Historical Temperature Trend (2000–Present)",
        yaxis_title="Max Temperature (°C)",
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_heatwave_calendar_heatmap(heatmap_df: pd.DataFrame):
    """Year x Month Heatwave Matrix Heatmap."""
    if heatmap_df.empty:
        st.info("No calendar heatmap data available.")
        return

    fig = px.imshow(
        heatmap_df,
        labels=dict(x="Month", y="Year", color="Heatwave Days"),
        x=heatmap_df.columns,
        y=heatmap_df.index,
        color_continuous_scale="YlOrRd",
        title="Heatwave Calendar Matrix (Heatwave Days per Month & Year)",
    )
    fig.update_layout(**DARK_LAYOUT, height=450)
    st.plotly_chart(fig, use_container_width=True)
