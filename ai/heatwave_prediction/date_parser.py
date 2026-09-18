from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from utils.constants import ERODE_TIMEZONE

MONTHS_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def get_current_today(timezone_str: str = ERODE_TIMEZONE) -> date:
    """Return current date in configured timezone (Erode / Asia/Kolkata)."""
    try:
        return datetime.now(ZoneInfo(timezone_str)).date()
    except Exception:
        return datetime.now().date()


def format_display_date(target_date: date) -> str:
    """Format date as '20 September 2026'."""
    day = target_date.day
    month_name = target_date.strftime("%B")
    year = target_date.year
    return f"{day} {month_name} {year}"


def parse_date_input(input_value: str | date, today_override: Optional[date] = None) -> Dict[str, Any]:
    """
    Parse a date input string or date object.
    Supports:
    - YYYY-MM-DD (2026-09-20)
    - DD-MM-YYYY (20-09-2026)
    - DD/MM/YYYY (20/09/2026)
    - Month DD, YYYY (September 20, 2026)
    - DD Month YYYY (20 September 2026)
    - Relative terms: today, tomorrow, yesterday, next week

    Returns a dict:
    {
        "valid": bool,
        "target_date": date,
        "iso_date": "YYYY-MM-DD",
        "formatted_date": "20 September 2026",
        "date_type": "TODAY" | "FUTURE" | "PAST",
        "raw_input": str,
        "error": Optional[str]
    }
    """
    today = today_override or get_current_today()

    if isinstance(input_value, date):
        target = input_value
        return _build_date_payload(target, today, str(input_value))

    text = str(input_value).strip()
    if not text:
        return {
            "valid": False,
            "target_date": today,
            "iso_date": today.isoformat(),
            "formatted_date": format_display_date(today),
            "date_type": "TODAY",
            "raw_input": text,
            "error": "Empty date string provided.",
        }

    lowered = text.lower().strip()

    # Relative terms
    if lowered in {"today", "now", "current"}:
        return _build_date_payload(today, today, text)
    if lowered == "tomorrow":
        return _build_date_payload(today + timedelta(days=1), today, text)
    if lowered == "yesterday":
        return _build_date_payload(today - timedelta(days=1), today, text)
    if lowered in {"next week", "nextweek", "in a week", "one week from now", "in 7 days"}:
        return _build_date_payload(today + timedelta(days=7), today, text)
    if lowered in {"next month", "in a month", "one month from now", "in 1 month"}:
        return _build_date_payload(_add_months(today, 1), today, text)
    if lowered in {"next year", "in a year", "one year from now", "same date next year"}:
        return _build_date_payload(_same_date_next_year(today), today, text)

    # Generic relative date phrases like "in 3 months", "6 months from now", "in 2 years"
    relative_match = re.match(r"^(?:in\s+)?(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+(?:from\s+now|later|hence)?$", lowered)
    if relative_match:
        count = int(relative_match.group(1))
        unit = relative_match.group(2)
        delta = timedelta(days=count) if unit.startswith("day") else timedelta(weeks=count) if unit.startswith("week") else timedelta(days=count * 30)
        if unit.startswith("month"):
            return _build_date_payload(_add_months(today, count), today, text)
        if unit.startswith("year"):
            return _build_date_payload(_same_date_next_year(today, offset_years=count), today, text)
        return _build_date_payload(today + delta, today, text)

    month_from_now_match = re.match(r"^(?:in\s+)?(\d+)\s+months?\s+from\s+now$", lowered)
    if month_from_now_match:
        return _build_date_payload(_add_months(today, int(month_from_now_match.group(1))), today, text)

    year_from_now_match = re.match(r"^(?:in\s+)?(\d+)\s+years?\s+from\s+now$", lowered)
    if year_from_now_match:
        return _build_date_payload(_same_date_next_year(today, offset_years=int(year_from_now_match.group(1))), today, text)

    # Clean punctuation surrounding text if user passed query like "prediction for 2026-09-20?"
    cleaned_text = re.sub(r"[?.,!]", "", text).strip()

    # 1. ISO format: YYYY-MM-DD or YYYY/MM/DD
    iso_match = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", cleaned_text)
    if iso_match:
        try:
            year, month, day = int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3))
            target = date(year, month, day)
            return _build_date_payload(target, today, text)
        except ValueError as exc:
            return _invalid_payload(text, f"Invalid calendar date: {exc}")

    # 2. Indian/European format: DD-MM-YYYY or DD/MM/YYYY
    dmy_match = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b", cleaned_text)
    if dmy_match:
        try:
            day, month, year = int(dmy_match.group(1)), int(dmy_match.group(2)), int(dmy_match.group(3))
            target = date(year, month, day)
            return _build_date_payload(target, today, text)
        except ValueError as exc:
            return _invalid_payload(text, f"Invalid calendar date: {exc}")

    # 3. Text format: "September 20, 2026", "20 September 2026", "Sep 20 2026"
    month_names_pattern = "|".join(MONTHS_MAP.keys())
    
    # Month DD, YYYY
    mdy_match = re.search(rf"\b({month_names_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*|\s+)(\d{{4}})\b", cleaned_text, re.IGNORECASE)
    if mdy_match:
        try:
            month = MONTHS_MAP[mdy_match.group(1).lower()]
            day = int(mdy_match.group(2))
            year = int(mdy_match.group(3))
            target = date(year, month, day)
            return _build_date_payload(target, today, text)
        except ValueError as exc:
            return _invalid_payload(text, f"Invalid date: {exc}")

    # DD Month YYYY
    dmy_text_match = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({month_names_pattern})(?:,\s*|\s+)(\d{{4}})\b", cleaned_text, re.IGNORECASE)
    if dmy_text_match:
        try:
            day = int(dmy_text_match.group(1))
            month = MONTHS_MAP[dmy_text_match.group(2).lower()]
            year = int(dmy_text_match.group(3))
            target = date(year, month, day)
            return _build_date_payload(target, today, text)
        except ValueError as exc:
            return _invalid_payload(text, f"Invalid date: {exc}")

    # Fallback standard strptime checks
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"):
        try:
            parsed_dt = datetime.strptime(cleaned_text, fmt).date()
            return _build_date_payload(parsed_dt, today, text)
        except ValueError:
            continue

    return _invalid_payload(
        text,
        "Unrecognized date format. Supported formats: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY, September 20, 2026, 20 September 2026, today, tomorrow, next week.",
    )


def _add_months(base: date, months: int) -> date:
    month_index = (base.year * 12) + (base.month - 1) + months
    year = month_index // 12
    month = (month_index % 12) + 1
    day = min(base.day, 28)
    return date(year, month, day)


def _same_date_next_year(base: date, offset_years: int = 1) -> date:
    target_year = base.year + offset_years
    try:
        return date(target_year, base.month, base.day)
    except ValueError:
        return date(target_year, base.month, 28)


def _build_date_payload(target: date, today: date, raw_input: str) -> Dict[str, Any]:
    if target == today:
        date_type = "TODAY"
    elif target > today:
        date_type = "FUTURE"
    else:
        date_type = "PAST"

    return {
        "valid": True,
        "target_date": target,
        "iso_date": target.isoformat(),
        "formatted_date": format_display_date(target),
        "date_type": date_type,
        "raw_input": raw_input,
        "error": None,
    }


def _invalid_payload(raw_input: str, error_msg: str) -> Dict[str, Any]:
    return {
        "valid": False,
        "target_date": None,
        "iso_date": None,
        "formatted_date": None,
        "date_type": "INVALID",
        "raw_input": raw_input,
        "error": error_msg,
    }
