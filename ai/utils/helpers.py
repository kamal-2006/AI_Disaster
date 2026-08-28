from __future__ import annotations

from datetime import datetime
import pandas as pd
from utils.constants import (
    COLOR_HIGH_RISK,
    COLOR_LOW_RISK,
    COLOR_MEDIUM_RISK,
    RISK_LEVEL_HIGH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
)


def get_risk_color(risk_level: str) -> str:
    """Return the HEX color string associated with a risk level."""
    if risk_level == RISK_LEVEL_HIGH:
        return COLOR_HIGH_RISK
    if risk_level == RISK_LEVEL_MEDIUM:
        return COLOR_MEDIUM_RISK
    return COLOR_LOW_RISK


def get_risk_badge_html(risk_level: str) -> str:
    """Generate an inline HTML badge for the risk level."""
    color = get_risk_color(risk_level)
    return (
        f'<span style="background-color: {color}22; color: {color}; '
        f'border: 1px solid {color}; padding: 4px 14px; border-radius: 12px; '
        f'font-weight: 700; font-size: 0.95rem; letter-spacing: 0.5px;">'
        f"{risk_level} RISK</span>"
    )


def format_timestamp(ts: datetime | pd.Timestamp | str | None) -> str:
    """Format a timestamp into a clean readable string."""
    if ts is None:
        return datetime.now().strftime("%Y-%m-%d %H:%M IST")
    if isinstance(ts, str):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            return ts
    if isinstance(ts, (datetime, pd.Timestamp)):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)


def format_temperature(val: float | None) -> str:
    """Format temperature value."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f} °C"


def format_percentage(val: float | None) -> str:
    """Format percentage value."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f} %"


def format_speed(val: float | None) -> str:
    """Format speed in km/h."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.1f} km/h"


def format_pressure(val: float | None) -> str:
    """Format VPD pressure in kPa."""
    if val is None or pd.isna(val):
        return "N/A"
    return f"{val:.2f} kPa"
