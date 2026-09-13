from __future__ import annotations

from datetime import date

import streamlit as st

from heatwave_prediction.predictor import predict_heatwave
from utils.constants import RISK_LEVEL_HIGH, RISK_LEVEL_MEDIUM


st.set_page_config(page_title="Date Heatwave Prediction", page_icon="🔥", layout="wide")

st.title("Date-Based Heatwave Prediction")
st.caption("Enter a date to retrieve Erode weather and run the trained heatwave model.")

selected_date = st.date_input(
    "Select date",
    value=date.today(),
    help="Choose today or a future date within the available Open-Meteo forecast range.",
)

if st.button("Predict Heatwave Risk", type="primary", use_container_width=True):
    with st.spinner("Retrieving weather and running the trained model..."):
        result = predict_heatwave(selected_date)

    if not result.get("available"):
        st.error(result["explanation"])
    else:
        date_label = date.fromisoformat(result["date"]).strftime("%d %B %Y")
        st.subheader(f"Heatwave Prediction for {date_label}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Location", result["location"])
        with col2:
            st.metric("Expected Temperature", f"{result['temperature']:.1f}°C")
        with col3:
            risk = result["risk"]
            st.metric("Heatwave Risk", risk.title())

        if risk == RISK_LEVEL_HIGH:
            st.error(result["prediction"])
        elif risk == RISK_LEVEL_MEDIUM:
            st.warning(result["prediction"])
        else:
            st.success(result["prediction"])

        st.info(result["explanation"])
        st.caption(f"Source: {result['weather_source']}")
        if result.get("probability") is not None:
            st.caption(f"Model probability: {result['probability'] * 100:.1f}%")
