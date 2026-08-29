from __future__ import annotations

import pandas as pd
import streamlit as st
from components.alerts import render_heatwave_alert_banner
from components.charts import plot_hourly_forecast
from components.disaster_assistant_ui import render_disaster_assistant_section
from components.explainability import render_explainability_section
from components.header import render_app_header
from components.metrics_cards import render_weather_metrics_grid
from components.recommendations import render_preparedness_section
from services.db_service import log_prediction_to_db
from services.prediction_service import get_realtime_prediction
from services.weather_api import fetch_erode_weather



def show_dashboard_page():
    with st.spinner("Updating live weather and running prediction pipeline..."):
        weather_payload = fetch_erode_weather()
        prediction_data = get_realtime_prediction()
        log_prediction_to_db(prediction_data)

    is_live = weather_payload.get("error") is None
    render_app_header(title="Heatwave Monitoring", is_live=is_live)

    current_weather = weather_payload.get("current", {})
    hourly_df = weather_payload.get("hourly_df", pd.DataFrame())


    # Check for prediction service error
    if "error" in prediction_data:
        st.error(f"⚠️ Model Prediction Notice: {prediction_data['error']}")

    # 1. Alert Banner
    render_heatwave_alert_banner(prediction_data)

    # 2. Key Metrics Grid
    st.markdown("### 📊 Live Weather & Heatwave Status")
    render_weather_metrics_grid(current_weather, prediction_data)

    st.divider()

    # 3. Forecast Trends Chart & XAI Factors
    st.markdown("### 📈 48-Hour Hourly Weather Trend")
    plot_hourly_forecast(hourly_df)

    st.divider()

    # 4. Explainable AI Factors
    render_explainability_section(prediction_data, top_n=8)

    st.divider()

    # 5. Quick Preparedness Recommendations
    render_preparedness_section(current_risk_level=prediction_data.get("risk_level", "LOW"))

    st.divider()

    # 6. AI Disaster Assistant Section
    render_disaster_assistant_section()


if __name__ == "__main__":

    show_dashboard_page()

