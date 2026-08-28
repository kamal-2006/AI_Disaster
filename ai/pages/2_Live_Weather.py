from __future__ import annotations

import pandas as pd
import streamlit as st
from components.charts import plot_hourly_forecast
from components.header import render_app_header
from services.weather_api import convert_hourly_to_daily_summary, fetch_erode_weather
from utils.helpers import format_percentage, format_pressure, format_speed, format_temperature


def show_live_weather_page():
    with st.spinner("Fetching live weather telemetry..."):
        weather_payload = fetch_erode_weather()

    is_live = weather_payload.get("error") is None
    render_app_header(title="Live Weather Observatory", is_live=is_live)

    current = weather_payload.get("current", {})
    hourly_df = weather_payload.get("hourly_df", pd.DataFrame())
    error = weather_payload.get("error")

    if error:
        st.warning("⚠️ Live weather data is temporarily unavailable. Please try again.")


    st.markdown("### 🌡️ Current Erode Weather Telemetry")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Temperature", format_temperature(current.get("temperature_2m")))
        st.metric("Apparent Temperature", format_temperature(current.get("apparent_temperature")))
        st.metric("Dew Point", format_temperature(current.get("dew_point_2m")))
    with col2:
        st.metric("Relative Humidity", format_percentage(current.get("relative_humidity_2m")))
        st.metric("Vapour Pressure Deficit", format_pressure(current.get("vapour_pressure_deficit")))
        st.metric("Cloud Cover", format_percentage(current.get("cloud_cover")))
    with col3:
        st.metric("Wind Speed", format_speed(current.get("wind_speed_10m")))
        st.metric("Wind Gusts", format_speed(current.get("wind_gusts_10m")))
        st.metric("Precipitation", f"{current.get('precipitation', 0.0):.1f} mm")

    st.caption(f"Last Updated: `{current.get('last_updated', 'N/A')}`")

    st.divider()

    st.markdown("### 📊 48-Hour Hourly Weather Trends")
    plot_hourly_forecast(hourly_df)

    st.divider()

    st.markdown("### 📅 Recent Daily Weather Summary")
    if not hourly_df.empty:
        daily_summary = convert_hourly_to_daily_summary(hourly_df)
        st.dataframe(daily_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Hourly history unavailable for daily summary rendering.")


if __name__ == "__main__":
    show_live_weather_page()

