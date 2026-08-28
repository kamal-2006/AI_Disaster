from __future__ import annotations

import pandas as pd
import streamlit as st
from common import load_pickle, resolve_path
from components.header import render_app_header

TRAINED_META_PATH = resolve_path("models", "training_metadata.pkl")
RESULTS_CSV = resolve_path("outputs", "model_results.csv")


def show_about_page():
    render_app_header(title="About System & Methodology")

    st.markdown(
        """
        ### 🌐 SafeGraph AI Framework Overview
        SafeGraph AI is an AI-powered disaster preparedness framework designed to predict heatwave events and communicate risk effectively for **Erode, Tamil Nadu**.

        #### 🎯 Key Technical Components
        * **Live Weather Telemetry**: Real-time atmospheric metrics fetched from Open-Meteo API.
        * **Time-Series ML Model**: 1-day ahead classification pipeline trained on historical climate baselines (2000–present).
        * **Explainable AI (XAI)**: Feature-level contribution analysis providing transparent explanations of prediction drivers.
        * **Disaster Risk Education**: Multi-persona emergency action guidelines tailored for students, teachers, administrators, and the public.
        """
    )

    st.divider()

    st.markdown("### 🤖 Model Performance & Evaluation Metrics")

    meta = load_pickle(TRAINED_META_PATH) if TRAINED_META_PATH.exists() else {}
    best_model_name = meta.get("best_model", "Logistic Regression")
    test_metrics = meta.get("test_metrics", {})

    st.markdown(f"**Selected Best Classifier**: `{best_model_name}`")

    if test_metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{test_metrics.get('Accuracy', 0.0)*100:.2f}%")
        with col2:
            st.metric("Precision", f"{test_metrics.get('Precision', 0.0)*100:.2f}%")
        with col3:
            st.metric("Recall", f"{test_metrics.get('Recall', 0.0)*100:.2f}%")
        with col4:
            st.metric("F1 Score", f"{test_metrics.get('F1', 0.0)*100:.2f}%")
        with col5:
            st.metric("ROC-AUC", f"{test_metrics.get('ROC_AUC', 0.0)*100:.2f}%")

    if RESULTS_CSV.exists():
        st.markdown("#### 📊 Model Comparison Matrix")
        results_df = pd.read_csv(RESULTS_CSV)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown(
        """
        ### 🏗️ System Dataflow Architecture
        ```text
        [ Open-Meteo API ] --------+
                                   |---> [ Live Telemetry & Daily Summaries ]
        [ Historical Climate CSV ] -+               |
                                                   v
                                   [ Feature Pipeline (35 Features) ]
                                                   |
                                                   v
                                   [ Trained ML Classifier (.pkl) ]
                                                   |
                                                   v
                                   [ Heatwave Prob, Risk & XAI Drivers ]
                                                   |
                                                   v
                                   [ SafeGraph AI Dashboard ]
        ```
        """
    )


if __name__ == "__main__":
    show_about_page()

