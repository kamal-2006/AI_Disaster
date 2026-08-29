from __future__ import annotations

import os
import re
import requests
import streamlit as st
from typing import Any, Dict, List, Optional, Tuple

from services.disaster_knowledge import (
    DISASTER_DISCLAIMER,
    DISASTER_KNOWLEDGE_BASE,
    ERODE_HELPLINES,
    KEYWORD_CATEGORY_MAP,
)
from services.prediction_service import get_realtime_prediction
from services.weather_api import fetch_erode_weather


# Comprehensive disaster & emergency topic indicators
DISASTER_TERMS = {
    "disaster", "emergency", "safety", "preparedness", "prepare", "heatwave", "heat wave",
    "extreme heat", "sunstroke", "heatstroke", "heat stroke", "temperature", "outside today",
    "precautions", "flood", "flooding", "waterlogging", "waterlog", "cyclone", "storm",
    "earthquake", "tremor", "quake", "landslide", "mudslide", "drought", "thunderstorm",
    "thunder", "lightning", "tsunami", "kit", "go-bag", "go bag", "first aid", "evacuation",
    "evacuate", "shelter", "warning", "helpline", "erode", "hazard", "risk", "rescue",
    "school prep", "student safety", "early warning", "after a disaster", "before a disaster",
}

# Off-topic topic indicators to catch general non-disaster queries
OFF_TOPIC_TERMS = {
    "cricket", "football", "soccer", "movie", "film", "actor", "actress", "song", "music",
    "recipe", "cooking", "restaurant", "politics", "election", "stock market", "shares",
    "crypto", "coding", "python code", "java", "javascript", "game", "gaming", "joke",
    "riddle", "poem", "essay", "fashion", "makeup", "car model", "iphone", "android",
}


def is_disaster_related_query(query: str) -> bool:
    """
    Determine if a user's natural language query is related to disaster preparedness,
    emergency response, safety, or weather risks.
    """
    q_clean = query.lower().strip()
    words = set(re.findall(r"\b\w+\b", q_clean))

    # Explicit check for common off-topic topics
    for word in words:
        if word in OFF_TOPIC_TERMS and not (words & DISASTER_TERMS):
            return False

    # Check if any disaster term appears in query
    if any(term in q_clean for term in DISASTER_TERMS):
        return True

    # Common question structures about safety, emergency, health during events
    if any(phrase in q_clean for phrase in ["what to do", "how to stay safe", "how to protect", "what should i", "items in", "early warning"]):
        return True

    return False


def identify_disaster_category(query: str) -> str:
    """Identify the primary disaster category from the query."""
    q_lower = query.lower()
    
    # Priority checks for specific query intent
    if any(k in q_lower for k in ["kit", "go bag", "go-bag", "first aid", "items"]):
        return "emergency_kit"
    
    if any(k in q_lower for k in ["erode", "outside today", "today in erode", "precautions today"]):
        return "heatwave"

    for category, keywords in KEYWORD_CATEGORY_MAP.items():
        if any(kw in q_lower for kw in keywords):
            return category

    return "general_prep"


def get_live_erode_context() -> Dict[str, Any]:
    """Retrieve live Erode weather and heatwave prediction risk level."""
    try:
        pred_data = get_realtime_prediction()
        weather_payload = fetch_erode_weather()
        current_w = weather_payload.get("current", {})
        
        return {
            "available": True,
            "risk_level": pred_data.get("risk_level", "LOW"),
            "probability": pred_data.get("probability", 0.0),
            "prediction_date": pred_data.get("target_date", "Today"),
            "temperature": current_w.get("temperature_2m", "N/A"),
            "max_temperature": current_w.get("max_temperature", "N/A"),
            "humidity": current_w.get("relative_humidity_2m", "N/A"),
            "apparent_temp": current_w.get("apparent_temperature", "N/A"),
        }
    except Exception as err:
        return {
            "available": False,
            "risk_level": "LOW",
            "probability": 0.0,
            "error": str(err),
        }


def get_llm_api_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Look for Gemini or OpenAI API keys in Streamlit secrets or environment variables.
    Returns (provider, api_key).
    """
    # 1. Check Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_key and hasattr(st, "secrets"):
        try:
            gemini_key = st.secrets.get("gemini", {}).get("api_key") or st.secrets.get("api_key")
        except Exception:
            pass

    if gemini_key:
        return "gemini", gemini_key

    # 2. Check OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key and hasattr(st, "secrets"):
        try:
            openai_key = st.secrets.get("openai", {}).get("api_key")
        except Exception:
            pass

    if openai_key:
        return "openai", openai_key

    return None, None


def call_llm_api(provider: str, api_key: str, prompt: str, system_instruction: str) -> Optional[str]:
    """Call LLM REST API (Gemini or OpenAI) via requests HTTP."""
    try:
        if provider == "gemini":
            # Gemini REST API Endpoint
            models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-pro"]
            for model_name in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
                payload = {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{system_instruction}\n\nUser Question: {prompt}"}],
                        }
                    ],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000},
                }
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    res_json = resp.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")

        elif provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
            }
            resp = requests.post(url, headers=headers, json=payload, timeout=10)
            if resp.status_code == 200:
                res_json = resp.json()
                choices = res_json.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "")

    except Exception:
        pass  # Gracefully fall back to knowledge base engine

    return None


def generate_grounded_fallback_response(category: str, query: str, live_context: Dict[str, Any]) -> str:
    """
    Generate a knowledge-grounded, structured response using the curated
    disaster knowledge base and active Erode weather context.
    """
    data = DISASTER_KNOWLEDGE_BASE.get(category, DISASTER_KNOWLEDGE_BASE["general_prep"])
    q_lower = query.lower()

    lines = []
    lines.append(f"### 🛡️ {data['title']}")
    lines.append(f"*{data['description']}*\n")

    # Inject dynamic Erode Live Context if relevant
    if category == "heatwave" or "erode" in q_lower or "today" in q_lower:
        if live_context.get("available"):
            risk = live_context["risk_level"]
            temp = live_context["max_temperature"]
            prob = live_context["probability"]
            
            risk_badge = f"🟢 **LOW RISK**" if risk == "LOW" else (f"🟠 **MEDIUM RISK**" if risk == "MEDIUM" else f"🔴 **HIGH RISK**")
            lines.append(
                f"> 📍 **Live Erode Conditions**: Model Risk Level: {risk_badge} | "
                f"Max Temp: **{temp}°C** | Heatwave Probability: **{prob * 100:.1f}%**\n"
            )

    # 1. Immediate Actions
    lines.append("#### 🚀 Immediate Actions")
    for act in data["immediate_actions"]:
        lines.append(f"- {act}")
    lines.append("")

    # 2. What to Avoid
    lines.append("#### ⚠️ What to Avoid")
    for av in data["what_to_avoid"]:
        lines.append(f"- {av}")
    lines.append("")

    # 3. Emergency / Student Specific Actions
    if "student" in q_lower or "school" in q_lower or "class" in q_lower:
        lines.append("#### 🎒 School & Student Safety Guidelines")
        for st_tip in data.get("school_student_tips", []):
            lines.append(f"- {st_tip}")
        lines.append("")

    lines.append("#### 🚑 Emergency Actions & Helplines")
    for em in data["emergency_actions"]:
        lines.append(f"- {em}")
    lines.append("")

    lines.append("**Key Erode Helpline Numbers:**")
    lines.append(f"- Medical Emergency: `{ERODE_HELPLINES['Medical Emergency / Ambulance']}`")
    lines.append(f"- District Emergency Control Room: `{ERODE_HELPLINES['District Emergency Control Room']}`")
    lines.append(f"- Water Supply Helpline: `{ERODE_HELPLINES['Corporation Water Supply']}`")
    lines.append("")
    lines.append(DISASTER_DISCLAIMER)

    return "\n".join(lines)


def get_disaster_assistant_response(user_query: str, session_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """
    Main service entry point for AI Disaster Assistant.
    Input: user_query string
    Returns: Dict containing formatted response, category, context, source, and disclaimer.
    """
    query_text = user_query.strip()
    if not query_text:
        return {
            "is_disaster": True,
            "response": "Please ask a question about disaster preparedness, emergency steps, or weather safety in Erode.",
            "source": "System",
            "category": "general_prep",
        }

    # 1. Intent Validation
    if not is_disaster_related_query(query_text):
        return {
            "is_disaster": False,
            "response": (
                "🤖 **SafeGraph AI Assistant Notice**\n\n"
                "I am specifically designed to assist with **disaster preparedness, emergency response, "
                "extreme weather safety, and Erode local risk advisories**.\n\n"
                "Your question appears to be outside of disaster preparedness. Please ask a disaster-related question, such as:\n"
                "- *'What should I do during a heatwave?'*\n"
                "- *'What precautions should I take today in Erode?'*\n"
                "- *'How can students stay safe during extreme heat?'*\n"
                "- *'What should be in an emergency kit?'*\n"
                "- *'What should I do during an earthquake or flood?'*"
            ),
            "source": "Domain Policy Guardrail",
            "category": "off_topic",
        }

    # 2. Identify Disaster Category & Live Context
    category = identify_disaster_category(query_text)
    live_context = get_live_erode_context()

    # 3. Check for configured LLM API Credentials
    provider, api_key = get_llm_api_credentials()

    if provider and api_key:
        system_instruction = (
            "You are the official SafeGraph AI Disaster Assistant for Erode, Tamil Nadu, India. "
            "Your task is to provide clear, accurate, actionable, and safety-focused disaster preparedness advice. "
            "Ground your answer in verified safety rules. "
            "Format responses using markdown with sections: Immediate Actions, What to Avoid, and Emergency Actions/Helplines. "
            f"Active Erode Heatwave Status: Risk Level = {live_context.get('risk_level')}, "
            f"Max Temperature = {live_context.get('max_temperature')}°C, Probability = {live_context.get('probability')}. "
            "Always include emergency helplines (108 Ambulance, 1077 Erode Control Room) and state that you do not replace official alerts."
        )

        llm_output = call_llm_api(provider, api_key, query_text, system_instruction)
        if llm_output:
            return {
                "is_disaster": True,
                "response": llm_output,
                "source": f"AI Model ({provider.capitalize()}) + Live Erode Context",
                "category": category,
                "live_context": live_context,
            }

    # 4. Fallback Knowledge Engine (No API key required)
    grounded_response = generate_grounded_fallback_response(category, query_text, live_context)
    return {
        "is_disaster": True,
        "response": grounded_response,
        "source": "SafeGraph AI Knowledge Base + Live Erode Context",
        "category": category,
        "live_context": live_context,
    }
