from __future__ import annotations

import os
import re
import requests
import streamlit as st
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from services.disaster_knowledge import (
    DISASTER_DISCLAIMER,
    DISASTER_KNOWLEDGE_BASE,
    ERODE_HELPLINES,
    KEYWORD_CATEGORY_MAP,
)
from services.prediction_service import get_realtime_prediction
from services.weather_api import fetch_erode_weather
from services.knowledge_graph import KNOWLEDGE_GRAPH
from services.weather_query import classify_weather_question, get_weather_result


logger = logging.getLogger(__name__)


# Comprehensive disaster & emergency topic indicators
DISASTER_TERMS = {
    "disaster", "emergency", "safety", "preparedness", "prepare", "heatwave", "heat wave",
    "extreme heat", "sunstroke", "heatstroke", "heat stroke", "temperature", "outside today",
    "precautions", "flood", "flooding", "waterlogging", "waterlog", "cyclone", "storm",
    "earthquake", "tremor", "quake", "landslide", "mudslide", "drought", "thunderstorm",
    "thunder", "lightning", "tsunami", "kit", "go-bag", "go bag", "first aid", "evacuation",
    "evacuate", "shelter", "warning", "helpline", "erode", "hazard", "risk", "rescue",
    "school prep", "student safety", "early warning", "after a disaster", "before a disaster",
    "heat exhaustion", "heatstroke symptoms", "dehydration", "dizziness", "nausea",
    "forest fire", "wildfire", "wild fire",
}

# Off-topic topic indicators to catch general non-disaster queries
OFF_TOPIC_TERMS = {
    "cricket", "football", "soccer", "movie", "film", "actor", "actress", "song", "music",
    "recipe", "cooking", "restaurant", "politics", "election", "stock market", "shares",
    "crypto", "coding", "python code", "java", "javascript", "game", "gaming", "joke",
    "riddle", "poem", "essay", "fashion", "makeup", "car model", "iphone", "android",
}


from heatwave_prediction.date_parser import parse_date_input


def is_disaster_related_query(query: str) -> bool:
    """
    Determine if a user's natural language query is related to disaster preparedness,
    emergency response, safety, weather risks, or date queries.
    """
    q_clean = query.lower().strip()
    words = set(re.findall(r"\b\w+\b", q_clean))

    # Check if query contains explicit date input
    if parse_date_input(query)["valid"]:
        return True

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

    category_matches = []
    for category, keywords in KEYWORD_CATEGORY_MAP.items():
        for keyword in keywords:
            if re.search(rf"(?<!\w){re.escape(keyword.lower())}s?(?!\w)", q_lower):
                category_matches.append((len(keyword), category))

    if category_matches:
        return max(category_matches, key=lambda match: (match[0], match[1] != "general_prep"))[1]

    if any(term in q_lower for term in ("forest fire", "wildfire", "wild fire")):
        return "forest_fire"

    return "general_prep"


INTENT_PATTERNS = {
    "symptoms": ("symptom", "signs", "warning signs", "feel dizzy", "nausea", "weakness"),
    "cause": ("why", "cause", "causes", "reason", "how does", "how do"),
    "immediate_action": ("what should i do", "what do i do", "immediately", "if .* enters", "during"),
    "preparedness": ("prepare", "preparedness", "before", "get ready", "home for", "what should be in", "what to pack", "items"),
    "prevention": ("prevent", "avoid", "reduce the risk", "protect"),
    "effects": ("effects", "impact", "happen during", "happens after", "risks"),
}

INTENT_SECTION_WEIGHTS = {
    "symptoms": {"emergency_actions": 8, "immediate_actions": 2},
    "cause": {"explanations": 9, "description": 2},
    "immediate_action": {"immediate_actions": 8, "emergency_actions": 6, "what_to_avoid": 5},
    "preparedness": {"immediate_actions": 6, "what_to_avoid": 3, "school_student_tips": 2},
    "prevention": {"what_to_avoid": 7, "immediate_actions": 3},
    "effects": {"explanations": 7, "emergency_actions": 3},
    "general": {"immediate_actions": 2, "emergency_actions": 2},
}

SECTION_LABELS = {
    "explanations": "Explanation",
    "immediate_actions": "Recommended actions",
    "what_to_avoid": "Important cautions",
    "emergency_actions": "Warning and emergency guidance",
    "school_student_tips": "School and student guidance",
    "description": "Background",
}


def detect_intent(query: str) -> str:
    """Classify the user's request without reducing it to a disaster category."""
    q_lower = query.lower()
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["symptoms"]):
        return "symptoms"
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["prevention"]):
        return "prevention"
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["preparedness"]):
        return "preparedness"
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["cause"]):
        return "cause"
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["immediate_action"]):
        return "immediate_action"
    if any(re.search(pattern, q_lower) for pattern in INTENT_PATTERNS["effects"]):
        return "effects"
    return "general"


def detect_topic(query: str, intent: str) -> str:
    q_lower = query.lower()
    if intent == "symptoms":
        return "warning_signs_and_symptoms"
    if intent == "cause":
        return "cause"
    if intent == "immediate_action":
        return "personal_safety" if any(term in q_lower for term in ("i am", "near me", "enters", "happening")) else "immediate_action"
    if intent == "preparedness" and any(term in q_lower for term in ("home", "house")):
        return "home_preparation"
    if intent == "preparedness":
        return "preparedness"
    if intent == "prevention":
        return "prevention"
    return "general_information"


def detect_urgency(query: str) -> str:
    if re.search(r"\b(i am|right now|currently|happening|near me|enters|immediately|active)\b", query.lower()):
        return "active_emergency"
    return "general"


def _query_terms(query: str) -> set[str]:
    stop_words = {
        "what", "are", "the", "is", "a", "an", "do", "does", "i", "my", "you", "how",
        "can", "should", "to", "of", "for", "during", "about", "if", "in", "on", "and",
        "or", "why", "happen", "happens", "would", "with", "from", "before", "after",
    }
    return {word for word in re.findall(r"[a-z0-9]+", query.lower()) if word not in stop_words and len(word) > 2}


def retrieve_question_relevant_knowledge(
    category: str,
    query: str,
    intent: str,
    topic: str = "",
    entities: Optional[List[str]] = None,
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Retrieve only knowledge nodes connected to the detected disaster node."""
    return KNOWLEDGE_GRAPH.retrieve(category, intent, topic, query, entities or [], limit)


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


def _weather_context(weather_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not weather_result:
        return {}
    weather = weather_result.get("weather", {}) or {}
    return {
        "location": weather_result.get("location"),
        "date": weather_result.get("date"),
        "date_type": weather_result.get("date_type"),
        "weather_source": weather_result.get("weather_source"),
        "forecast_range": weather_result.get("forecast_range"),
        "temperature": weather_result.get("temperature"),
        "humidity": weather.get("mean_relative_humidity", weather.get("relative_humidity_2m")),
        "apparent_temperature": weather.get("mean_apparent_temperature", weather.get("apparent_temperature")),
        "risk_level": weather_result.get("risk_level"),
        "probability": weather_result.get("probability"),
        "model_name": weather_result.get("prediction", {}).get("model_name") if weather_result.get("prediction") else weather_result.get("model_name"),
        "error": weather_result.get("error"),
    }


def _attach_debug_payload(
    result: Dict[str, Any],
    understanding: Dict[str, Any],
    retrieved_knowledge: List[Dict[str, Any]],
    final_prompt: str,
    weather_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach internal pipeline evidence only when explicitly enabled for development."""
    if os.getenv("SAFEGRAPH_AI_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return result

    result["debug"] = {
        "original_question": understanding["question"],
        "question_understanding": understanding,
        "retrieved_graph_nodes": [
            {key: item.get(key) for key in ("node_id", "node_label", "category", "section", "topic", "score", "text")}
            for item in retrieved_knowledge
        ],
        "retrieved_relationships": [
            relationship
            for item in retrieved_knowledge
            for relationship in item.get("relationships", [])
        ],
        "relevant_context": result.get("retrieved_context", ""),
        "weather": _weather_context(weather_result),
        "model_features": (
            weather_result.get("prediction", {}).get("feature_row").to_dict(orient="records")[0]
            if weather_result and weather_result.get("prediction", {}).get("feature_row") is not None
            else weather_result.get("feature_row").to_dict(orient="records")[0]
            if weather_result and weather_result.get("feature_row") is not None
            else {}
        ),
        "heatwave_prediction": weather_result.get("prediction_class") if weather_result else None,
        "risk_level": weather_result.get("risk_level") if weather_result else None,
        "final_llm_prompt": final_prompt,
        "final_answer": result["response"],
    }
    return result


def generate_grounded_fallback_response(
    category: str,
    query: str,
    intent: str,
    retrieved_knowledge: List[Dict[str, Any]],
    live_context: Dict[str, Any],
) -> str:
    """
    Generate a knowledge-grounded, structured response using the curated
    disaster knowledge base and active Erode weather context.
    """
    data = DISASTER_KNOWLEDGE_BASE.get(category)
    q_lower = query.lower()

    if not data or not retrieved_knowledge:
        return (
            "I couldn't find enough specific information in my disaster knowledge base "
            "to answer that question accurately. Please follow current instructions from "
            "local emergency authorities."
        )

    lines = [f"### 🛡️ {data['title']}"]
    intent_headings = {
        "symptoms": "### Direct answer: warning signs",
        "cause": "### Explanation",
        "immediate_action": "### Direct answer: immediate actions",
        "preparedness": "### Direct answer: preparedness",
        "prevention": "### Direct answer: prevention",
        "effects": "### Direct answer: effects and risks",
        "general": "### Relevant guidance",
    }
    lines.append(intent_headings[intent])

    for item in retrieved_knowledge:
        label = SECTION_LABELS.get(item["section"], "Relevant guidance")
        lines.append(f"- **{label}:** {item['text']}")
    lines.append("")

    # Inject dynamic Erode Live Context if relevant
    if "erode" in q_lower or "today" in q_lower:
        if live_context.get("available"):
            risk = live_context["risk_level"]
            temp = live_context["max_temperature"]
            prob = live_context["probability"]
            
            risk_badge = f"🟢 **LOW RISK**" if risk == "LOW" else (f"🟠 **MEDIUM RISK**" if risk == "MEDIUM" else f"🔴 **HIGH RISK**")
            lines.append(
                f"> 📍 **Live Erode Conditions**: Model Risk Level: {risk_badge} | "
                f"Max Temp: **{temp}°C** | Heatwave Probability: **{prob * 100:.1f}%**\n"
            )

    # Add only directly requested supporting guidance, rather than the full category checklist.
    if intent in {"immediate_action", "preparedness", "prevention"} and "school" in q_lower:
        lines.append("#### 🎒 School & Student Safety Guidelines")
        for st_tip in data.get("school_student_tips", []):
            lines.append(f"- {st_tip}")
        lines.append("")

    if intent in {"symptoms", "immediate_action"}:
        lines.append("For an active emergency, call **108** or the Erode District Control Room at **1077**.")
    lines.append(DISASTER_DISCLAIMER)

    return "\n".join(lines)


def generate_weather_fallback_response(weather_result: Dict[str, Any]) -> str:
    """Format API and model results without allowing generated weather values."""
    if not weather_result.get("available"):
        return weather_result.get("error", "Weather data or heatwave prediction is unavailable for the requested date.")

    if weather_result.get("user_response"):
        return weather_result["user_response"]
    if isinstance(weather_result.get("prediction"), dict) and weather_result["prediction"].get("user_response"):
        return weather_result["prediction"]["user_response"]

    date_type = weather_result.get("date_type")
    temperature = weather_result.get("temperature")
    risk_level = weather_result.get("risk_level", "unavailable")
    probability = weather_result.get("probability")
    temperature_text = f"{float(temperature):.1f}°C" if temperature is not None else "unavailable"
    risk_text = str(risk_level).title()

    if date_type == "today":
        lines = [
            "## Heatwave Prediction",
            "",
            "Location: Erode",
            "Date: Today",
            "",
            f"Current Temperature: {temperature_text}",
            f"Heatwave Risk: {risk_text}",
            "",
            f"Prediction: Current weather conditions indicate a {risk_text.lower()} heatwave risk.",
        ]
    elif date_type == "future":
        formatted_date = datetime.strptime(weather_result["date"], "%Y-%m-%d").strftime("%d %B %Y")
        lines = [
            "## Heatwave Prediction",
            "",
            "Location: Erode",
            f"Date: {formatted_date}",
            "",
            f"Forecast Temperature: {temperature_text}",
            f"Heatwave Risk: {risk_text}",
            "",
            f"Prediction: Heatwave conditions are {'likely' if str(risk_level).upper() in ('HIGH', '1') else 'unlikely'} based on the forecast weather conditions and the trained heatwave model.",
            "",
            "Data Source:",
            "Open-Meteo Forecast + SafeGraph AI Heatwave Model",
        ]
    else:
        formatted_date = datetime.strptime(weather_result["date"], "%Y-%m-%d").strftime("%d %B %Y")
        lines = [
            "## Heatwave Prediction (Historical)",
            "",
            "Location: Erode",
            f"Date: {formatted_date}",
            "",
            f"Recorded Temperature: {temperature_text}",
            f"Heatwave Risk: {risk_text}",
        ]

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

    # Resolve short follow-ups against the most recent user question.
    user_messages = [message.get("content", "") for message in (session_history or []) if message.get("role") == "user"]
    prior_user_query = user_messages[-1] if user_messages else ""
    if prior_user_query.strip() == query_text and len(user_messages) > 1:
        prior_user_query = user_messages[-2]
    contextual_query = f"{prior_user_query} {query_text}".strip() if prior_user_query else query_text

    # 1. Intent Validation
    if not is_disaster_related_query(contextual_query):
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

    # 2. Identify disaster, intent, and ranked question-specific context.
    category = identify_disaster_category(contextual_query)
    weather_request = classify_weather_question(query_text)
    if weather_request and category == "general_prep":
        category = "heatwave"
    graph_intent = detect_intent(query_text)
    intent = weather_request["intent"] if weather_request else graph_intent
    topic = detect_topic(query_text, graph_intent)
    urgency = detect_urgency(query_text)
    entities = sorted(_query_terms(query_text))
    understanding = {
        "disaster": category,
        "intent": intent,
        "topic": topic,
        "urgency": urgency,
        "entities": entities,
        "question": query_text,
    }
    if weather_request:
        understanding.update({
            "location": weather_request["location"],
            "requested_date": weather_request["date"],
            "date_type": weather_request["date_type"],
            "weather_intent": weather_request["intent"],
        })

    weather_result = get_weather_result(weather_request) if weather_request else None

    if category not in DISASTER_KNOWLEDGE_BASE:
        disaster_name = category.replace("_", " ")
        response = (
            "I recognized this as a disaster-related question, but the current SafeGraph AI "
            f"Knowledge Graph has no {disaster_name} knowledge for it. I cannot provide a reliable "
            "answer from this knowledge source yet; follow local emergency authorities during an active emergency."
        )
        return _attach_debug_payload({
            "is_disaster": True,
            "response": response,
            "source": "SafeGraph AI Knowledge Graph (no matching disaster data)",
            "category": category,
            "understanding": understanding,
            "retrieved_context": "",
        }, understanding, [], "", weather_result)

    retrieval_query = contextual_query
    graph_only_weather = weather_request and weather_request["intent"] in {"current_weather", "future_weather", "historical_weather"}
    retrieved_knowledge = [] if graph_only_weather else retrieve_question_relevant_knowledge(category, retrieval_query, graph_intent, topic, entities)
    logger.debug("Original question: %s", query_text)
    logger.debug("Detected disaster: %s", category)
    logger.debug("Detected intent: %s", intent)
    logger.debug("Detected topic: %s", topic)
    logger.debug("Detected urgency: %s", urgency)
    logger.debug("Extracted entities: %s", entities)
    logger.debug("Retrieval query: %s", retrieval_query)
    logger.debug("Retrieved knowledge and ranking: %s", retrieved_knowledge)

    live_context = {"available": False}
    final_context = "\n".join(
        f"[Graph node: {item['node_id']} | Topic: {item['topic']} | Score: {item['score']}] {item['text']}"
        for item in retrieved_knowledge
    )
    logger.debug("Final context: %s", final_context)

    # 3. Check for configured LLM API Credentials
    provider, api_key = get_llm_api_credentials()
    final_llm_prompt = ""

    if provider and api_key:
        final_llm_prompt = (
            "You are SafeGraph AI, an explainable disaster-preparedness assistant. "
            "The SafeGraph AI Knowledge Graph is the primary factual source. "
            "Answer the user's exact question using only the retrieved graph context; "
            "do not answer from generic disaster knowledge or unrelated graph nodes. "
            "If the context is insufficient, say so instead of inventing facts. "
            "If the user asks why, explain the cause; for how, explain the procedure; "
            "for symptoms, list symptoms; for immediate action, prioritize safety and evacuation. "
            "Explain the answer clearly and omit retrieved facts that do not help answer the question.\n\n"
            f"USER QUESTION:\n{query_text}\n\n"
            f"QUESTION UNDERSTANDING:\n{understanding}\n\n"
            f"KNOWLEDGE GRAPH CONTEXT:\n{final_context or 'No sufficiently relevant graph knowledge was found.'}\n"
            f"WEATHER API AND MODEL RESULT (authoritative numbers; do not modify):\n{_weather_context(weather_result) or 'Not a weather question.'}"
        )

        llm_output = call_llm_api(provider, api_key, query_text, final_llm_prompt)
        if llm_output:
            logger.debug("Generated response: %s", llm_output)
            llm_source = f"SafeGraph AI Knowledge Graph + AI Model ({provider.capitalize()})"
            if weather_result:
                llm_source = f"{weather_result.get('weather_source', 'Weather API')} + trained heatwave model + AI Model ({provider.capitalize()})"
            return _attach_debug_payload({
                "is_disaster": True,
                "response": llm_output,
                "source": llm_source,
                "category": category,
                "live_context": live_context,
                "understanding": understanding,
                "retrieved_context": final_context,
                "weather_result": weather_result,
            }, understanding, retrieved_knowledge, final_llm_prompt, weather_result)

    # 4. Fallback Knowledge Engine (No API key required)
    grounded_response = generate_weather_fallback_response(weather_result) if weather_result else generate_grounded_fallback_response(category, query_text, graph_intent, retrieved_knowledge, live_context)
    logger.debug("Generated response: %s", grounded_response)
    fallback_source = "SafeGraph AI Knowledge Graph (local)"
    if weather_result:
        fallback_source = f"{weather_result.get('weather_source', 'Weather API')} + trained heatwave model"
    return _attach_debug_payload({
        "is_disaster": True,
        "response": grounded_response,
        "source": fallback_source,
        "category": category,
        "live_context": live_context,
        "understanding": understanding,
        "retrieved_context": final_context,
        "weather_result": weather_result,
    }, understanding, retrieved_knowledge, final_llm_prompt, weather_result)
