from __future__ import annotations

import streamlit as st
from utils.constants import (
    COLOR_HIGH_RISK,
    COLOR_LOW_RISK,
    COLOR_MEDIUM_RISK,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)


def render_heatwave_alert_banner(prediction_data: dict[str, object]):
    """
    Render prominent color-coded heatwave alert banner based on risk level.
    """
    risk_level = prediction_data.get("risk_level", RISK_LEVEL_LOW)
    prob = prediction_data.get("probability", 0.0)
    target_date = prediction_data.get("target_date", "Tomorrow")

    if risk_level == RISK_LEVEL_HIGH:
        st.markdown(
            f"""
            <div style="background-color: #7F1D1D25; border: 2px solid {COLOR_HIGH_RISK}; border-radius: 12px; padding: 20px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h3 style="color: {COLOR_HIGH_RISK}; margin: 0; display: flex; align-items: center; gap: 10px;">
                        🚨 EMERGENCY HEATWAVE RED ALERT - HIGH RISK DETECTED
                    </h3>
                    <span style="background-color: {COLOR_HIGH_RISK}; color: white; padding: 4px 14px; border-radius: 20px; font-weight: 800; font-size: 0.9rem;">
                        {prob*100:.1f}% RISK PROBABILITY
                    </span>
                </div>
                <p style="color: #FECACA; font-size: 1rem; margin-top: 10px; margin-bottom: 0;">
                    <b>Model Target Date: {target_date} | Location: Erode, Tamil Nadu</b><br>
                    Extreme temperature accumulation and vapor pressure deficit levels exceed critical safety thresholds. 
                    Immediate emergency disaster preparedness protocols must be activated for schools, public facilities, and outdoor workers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif risk_level == RISK_LEVEL_MEDIUM:
        st.markdown(
            f"""
            <div style="background-color: #78350F25; border: 2px solid {COLOR_MEDIUM_RISK}; border-radius: 12px; padding: 18px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h3 style="color: {COLOR_MEDIUM_RISK}; margin: 0; display: flex; align-items: center; gap: 10px;">
                        ⚠️ HEATWAVE ADVISORY - MEDIUM RISK DETECTED
                    </h3>
                    <span style="background-color: {COLOR_MEDIUM_RISK}; color: black; padding: 4px 14px; border-radius: 20px; font-weight: 800; font-size: 0.9rem;">
                        {prob*100:.1f}% RISK PROBABILITY
                    </span>
                </div>
                <p style="color: #FDE68A; font-size: 0.95rem; margin-top: 8px; margin-bottom: 0;">
                    <b>Model Target Date: {target_date} | Location: Erode, Tamil Nadu</b><br>
                    Weather conditions indicate moderate heat stress. Precautionary measures (hydration, shaded activities, PE class schedule adjustments) should be enforced.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background-color: #064E3B25; border: 1px solid {COLOR_LOW_RISK}; border-radius: 12px; padding: 16px; margin-bottom: 24px;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <h4 style="color: {COLOR_LOW_RISK}; margin: 0; display: flex; align-items: center; gap: 8px;">
                        ✅ NORMAL WEATHER CONDITIONS - LOW HEAT RISK
                    </h4>
                    <span style="background-color: {COLOR_LOW_RISK}33; color: {COLOR_LOW_RISK}; border: 1px solid {COLOR_LOW_RISK}; padding: 3px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                        {prob*100:.1f}% PROBABILITY
                    </span>
                </div>
                <p style="color: #A7F3D0; font-size: 0.9rem; margin-top: 6px; margin-bottom: 0;">
                    <b>Target Date: {target_date} | Location: Erode, Tamil Nadu</b><br>
                    No immediate heatwave warning for tomorrow. Maintain standard summer hydration guidelines.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
