from __future__ import annotations

import streamlit as st
from utils.helpers import (
    format_percentage,
    format_pressure,
    format_speed,
    format_temperature,
    get_risk_badge_html,
    get_risk_color,
)


def render_weather_metrics_grid(current_weather: dict[str, object], prediction_data: dict[str, object] | None = None):
    """
    Render 8 key metric cards in a clean layout:
    [Temperature] [Humidity] [Apparent Temp] [Wind Speed]
    [VPD] [Precipitation] [Heatwave Prob] [Risk Level Badge]
    """
    temp = current_weather.get("temperature_2m")
    humidity = current_weather.get("relative_humidity_2m")
    apparent_temp = current_weather.get("apparent_temperature")
    wind_speed = current_weather.get("wind_speed_10m")
    vpd = current_weather.get("vapour_pressure_deficit")
    precip = current_weather.get("precipitation", 0.0)
    updated_at = current_weather.get("last_updated", "N/A")

    prob = prediction_data.get("probability", 0.0) if prediction_data else 0.0
    risk_level = prediction_data.get("risk_level", "LOW") if prediction_data else "LOW"
    risk_color = get_risk_color(risk_level)

    # 1. Top row: Weather metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌡️ Current Temp", format_temperature(temp), delta=f"Feels like {format_temperature(apparent_temp)}")
    with col2:
        st.metric("💧 Relative Humidity", format_percentage(humidity))
    with col3:
        st.metric("🌡️ Apparent Temp", format_temperature(apparent_temp))
    with col4:
        st.metric("💨 Wind Speed", format_speed(wind_speed), delta=f"Gusts {format_speed(current_weather.get('wind_gusts_10m'))}")

    # 2. Second row: VPD, Precip, Heatwave Probability, Risk Badge
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("🌤️ Vapour Pressure Deficit", format_pressure(vpd))
    with col6:
        st.metric("🌧️ Precipitation", f"{precip:.1f} mm")
    with col7:
        st.metric("🎯 Heatwave Probability", f"{prob * 100:.1f}%")
    with col8:
        st.markdown(
            f"""
            <div style="background-color: #1E293B; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #334155;">
                <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 600;">Erode Risk Status</div>
                <div style="margin-top: 6px;">{get_risk_badge_html(risk_level)}</div>
                <div style="font-size: 0.7rem; color: #64748B; margin-top: 4px;">Updated: {updated_at}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
