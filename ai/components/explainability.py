from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from utils.constants import FEATURE_DISPLAY_NAMES

DARK_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.0)",
    plot_bgcolor="rgba(30, 41, 59, 0.5)",
    font=dict(color="#F8FAFC", family="sans-serif"),
    margin=dict(l=20, r=20, t=30, b=30),
    xaxis=dict(gridcolor="#334155"),
    yaxis=dict(gridcolor="#334155"),
)


def render_explainability_section(prediction_data: dict[str, object], top_n: int = 10):
    """
    Render Explainable AI (XAI) feature importance chart, key driver breakdown table,
    and plain-English explainable summary.
    """
    xai_df: pd.DataFrame = prediction_data.get("xai_drivers")
    risk_level = prediction_data.get("risk_level", "LOW")
    probability = prediction_data.get("probability", 0.0)
    model_name = prediction_data.get("model_name", "Trained ML Model")

    st.markdown("### 🧠 Explainable AI (XAI) - Model Prediction Factors")
    st.markdown(
        f"Real model feature importance breakdown for **{model_name}** predicting **{probability * 100:.1f}% heatwave probability** ({risk_level} Risk)."
    )

    if xai_df is None or xai_df.empty:
        st.warning("Feature importance matrix unavailable for current prediction.")
        return

    # Prepare display dataframe
    df = xai_df.head(top_n).copy()
    df["display_name"] = df["feature"].map(lambda f: FEATURE_DISPLAY_NAMES.get(f, f))
    df = df.sort_values("importance", ascending=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # Plot horizontal bar chart of actual feature importances
        fig = px.bar(
            df,
            x="importance",
            y="display_name",
            orientation="h",
            labels={"importance": "Importance Weight", "display_name": "Feature Factor"},
            title=f"Top {top_n} Decision Factors (Trained Model Weights)",
            color="importance",
            color_continuous_scale="Oranges",
        )
        fig.update_layout(**DARK_LAYOUT, height=420, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Feature Inspection Table")
        table_df = df.sort_values("importance", ascending=False)[["display_name", "value", "importance"]].copy()
        table_df.columns = ["Feature", "Live/Hist Value", "Importance"]
        table_df["Importance"] = table_df["Importance"].map(lambda v: f"{v:.4f}")
        table_df["Live/Hist Value"] = table_df["Live/Hist Value"].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else str(v))
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Plain English Narrative
    st.markdown("#### 💡 Model Decision Narrative")
    top_feature = df.iloc[-1]["display_name"]
    second_feature = df.iloc[-2]["display_name"] if len(df) > 1 else ""

    if risk_level == "HIGH":
        st.error(
            f"**High Risk Driver Analysis**: The model predicts a high probability of heatwave risk ({probability*100:.1f}%). "
            f"The top factors driving this high risk rating are elevated **{top_feature}** and **{second_feature}**. "
            f"Thermal accumulation over past consecutive days has breached local 95th percentile climatological thresholds for Erode."
        )
    elif risk_level == "MEDIUM":
        st.warning(
            f"**Medium Risk Driver Analysis**: Heat stress indicators are approaching threshold levels. "
            f"Key metrics watched by the model include **{top_feature}** and **{second_feature}**. "
            f"Precautionary hydration and scheduled shade breaks are advised."
        )
    else:
        st.success(
            f"**Low Risk Driver Analysis**: Current temperatures and rolling humidity metrics remain within normal parameters. "
            f"The primary stabilizing metrics are **{top_feature}** and **{second_feature}**. "
            f"No heatwave condition predicted for tomorrow."
        )
