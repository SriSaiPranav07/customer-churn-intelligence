"""Utility functions, configuration constants, and formatting helpers."""
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_intelligence")

# Path constants
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"

# Risk Threshold Constants
# Calibrated for business retention operations
RISK_THRESHOLD_HIGH = 0.60
RISK_THRESHOLD_MEDIUM = 0.35
OPTIMAL_DECISION_THRESHOLD = 0.42

# Action Statuses
ACTION_STATUS_OPTIONS = [
    "Planned",
    "Contacted",
    "In Progress",
    "Resolved",
    "Declined",
    "Not Applicable"
]

# Color constants for Streamlit UI
COLOR_HIGH_RISK = "#EF4444"     # Red
COLOR_MEDIUM_RISK = "#F59E0B"   # Amber/Orange
COLOR_LOW_RISK = "#10B981"      # Green
COLOR_PRIMARY = "#2563EB"       # Modern Blue
COLOR_SECONDARY = "#64748B"     # Slate Gray


def get_risk_category(prob: float, high_thresh: float = RISK_THRESHOLD_HIGH, med_thresh: float = RISK_THRESHOLD_MEDIUM) -> str:
    """Classifies a churn probability into risk categories."""
    if prob >= high_thresh:
        return "High Risk"
    elif prob >= med_thresh:
        return "Medium Risk"
    return "Low Risk"


def get_risk_badge_color(category: str) -> str:
    """Returns hex color code for given risk tier."""
    if category == "High Risk":
        return COLOR_HIGH_RISK
    elif category == "Medium Risk":
        return COLOR_MEDIUM_RISK
    return COLOR_LOW_RISK


def format_currency(val: float) -> str:
    """Format float into USD currency representation."""
    try:
        return f"${val:,.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def format_percentage(val: float) -> str:
    """Format float (0-1) into percentage string."""
    try:
        return f"{val * 100:.1f}%"
    except (ValueError, TypeError):
        return "0.0%"
