"""
Heatwave Prediction Module for Erode.
Date-aware forecast weather retrieval, feature engineering, and trained ML inference.
"""

from .date_parser import parse_date_input
from .predictor import predict_heatwave

__all__ = ["predict_heatwave", "parse_date_input"]