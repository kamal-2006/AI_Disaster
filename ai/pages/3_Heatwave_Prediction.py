from __future__ import annotations

import pandas as pd
import streamlit as st
from components.alerts import render_heatwave_alert_banner
from components.explainability import render_explainability_section
from components.header import render_app_header
from services.prediction_service import get_realtime_prediction, predict_custom_scenario
from utils.constants import FEATURE_DISPLAY_NAMES
from utils.helpers import get_risk_badge_html


def show_prediction_page():
    render_app_header(title="Heatwave Prediction & Simulator")

    with st.spinner("Running heatwave prediction pipeline..."):
        prediction_data = get_realtime_prediction()

    if "error" in prediction_data:
        st.error(f"⚠️ Model Service Notice: {prediction_data['error']}")

    prob = prediction_data.get("probability", 0.0)
    risk_level = prediction_data.get("risk_level", "LOW")
    target_date = prediction_data.get("target_date", "N/A")
    model_name = prediction_data.get("model_name", "Logistic Regression")
    feature_row = prediction_data.get("feature_row", pd.DataFrame())

    # 1. Prediction summary hero card
    st.markdown("### 🎯 Live Model Prediction")
    render_heatwave_alert_banner(prediction_data)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Target Date", target_date)
    with col2:
        st.metric("Heatwave Probability", f"{prob * 100:.1f}%")
    with col3:
        st.markdown("##### Risk Level")
        st.markdown(get_risk_badge_html(risk_level), unsafe_allow_html=True)
    with col4:
        st.metric("Model Classifier", model_name)

    st.divider()

    # 2. XAI Explainability
    render_explainability_section(prediction_data, top_n=10)

    st.divider()

    # 3. Interactive Scenario Simulator
    st.markdown("### 🎛️ Interactive Heatwave Risk Simulator")
    st.caption("Simulate custom weather conditions to see how the model updates risk predictions in real time.")

    col_sim1, col_sim2 = st.columns(2)

    with col_sim1:
        sim_max_temp = st.slider("Simulated Max Temp (°C)", min_value=30.0, max_value=46.0, value=38.5, step=0.5)
        sim_mean_temp = st.slider("Simulated Mean Temp (°C)", min_value=22.0, max_value=40.0, value=32.0, step=0.5)
        sim_humidity = st.slider("Simulated Relative Humidity (%)", min_value=10.0, max_value=90.0, value=45.0, step=5.0)

    with col_sim2:
        sim_vpd = st.slider("Simulated Vapour Pressure Deficit (kPa)", min_value=0.5, max_value=6.0, value=3.2, step=0.1)
        sim_wind = st.slider("Simulated Wind Speed (km/h)", min_value=2.0, max_value=40.0, value=12.0, step=1.0)

    if st.button("🚀 Run Risk Simulation", type="primary", use_container_width=True):
        sim_res = predict_custom_scenario(
            max_temp=sim_max_temp,
            mean_temp=sim_mean_temp,
            humidity=sim_humidity,
            vpd=sim_vpd,
            wind_speed=sim_wind,
        )

        sim_prob = sim_res["probability"]
        sim_risk = sim_res["risk_level"]

        st.markdown("#### 🧪 Simulation Results")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Target Date", sim_res["target_date"])
        with res_col2:
            st.metric("Simulated Probability", f"{sim_prob * 100:.1f}%", delta=f"{((sim_prob - prob) * 100):+.1f}% vs Live")
        with res_col3:
            st.markdown("##### Simulated Risk Badge")
            st.markdown(get_risk_badge_html(sim_risk), unsafe_allow_html=True)

        render_explainability_section(sim_res, top_n=6)

    st.divider()

    # 4. Feature Vector Inspector
    if isinstance(feature_row, pd.DataFrame) and not feature_row.empty:
        with st.expander("📄 Prepared Model Input Features"):
            df_feat = feature_row.T.reset_index()
            df_feat.columns = ["Feature Column", "Prepared Value"]
            df_feat["Feature Name"] = df_feat["Feature Column"].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
            df_feat = df_feat[["Feature Name", "Prepared Value"]]
            st.dataframe(df_feat, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    show_prediction_page()

