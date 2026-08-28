from __future__ import annotations

from datetime import datetime
import os
import streamlit as st

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


def get_mongo_client():
    """
    Safely initialize MongoDB client using secrets.toml or environment variables.
    Returns (client, db) or (None, None).
    """
    if not PYMONGO_AVAILABLE:
        return None, None

    mongo_uri = None
    db_name = "safegraph_ai"

    # 1. Check Streamlit secrets
    try:
        if "mongodb" in st.secrets:
            mongo_uri = st.secrets["mongodb"].get("uri")
            db_name = st.secrets["mongodb"].get("database", "safegraph_ai")
    except Exception:
        pass

    # 2. Check environment variable
    if not mongo_uri:
        mongo_uri = os.getenv("MONGODB_URI")

    if not mongo_uri:
        return None, None

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
        # Quick server ping check
        client.admin.command("ping")
        db = client[db_name]
        return client, db
    except Exception:
        return None, None


def log_prediction_to_db(prediction_data: dict[str, object] | None) -> bool:
    """Save heatwave prediction result to MongoDB (if available)."""
    if not isinstance(prediction_data, dict):
        return False

    client = None
    try:
        client, db = get_mongo_client()
        if db is None:
            return False

        record = {
            "timestamp": datetime.now().isoformat(),
            "target_date": prediction_data.get("target_date"),
            "probability": prediction_data.get("probability"),
            "risk_level": prediction_data.get("risk_level"),
            "prediction_class": prediction_data.get("prediction_class"),
            "model_name": prediction_data.get("model_name"),
            "location": "Erode, Tamil Nadu",
        }
        db["heatwave_predictions"].insert_one(record)
        return True
    except Exception:
        return False
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


def get_db_status() -> dict[str, object]:
    """Check MongoDB availability status for UI indicator."""
    client = None
    try:
        client, db = get_mongo_client()
        if db is not None:
            return {"connected": True, "message": "Storage Active"}
        return {"connected": False, "message": "History storage unavailable"}
    except Exception:
        return {"connected": False, "message": "History storage unavailable"}
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass


