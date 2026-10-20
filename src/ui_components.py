"""Shared UI components, styling injection, and layout utilities for Streamlit."""
import streamlit as st
import pandas as pd
from typing import Tuple, Dict, Any

from src.utils import (
    COLOR_PRIMARY, COLOR_HIGH_RISK, COLOR_MEDIUM_RISK, COLOR_LOW_RISK,
    get_risk_badge_color, format_currency, format_percentage
)


def inject_custom_css():
    """Injects modern, enterprise CSS styling into the Streamlit dashboard."""
    st.markdown("""
        <style>
        /* Base typography & clean background */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Metric card styling */
        .metric-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
        }
        .metric-title {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 0.4rem;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
        }
        .metric-subtitle {
            font-size: 0.8rem;
            color: #94A3B8;
            margin-top: 0.35rem;
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-align: center;
        }
        .badge-high {
            background-color: #FEE2E2;
            color: #DC2626;
            border: 1px solid #FCA5A5;
        }
        .badge-med {
            background-color: #FEF3C7;
            color: #D97706;
            border: 1px solid #FCD34D;
        }
        .badge-low {
            background-color: #D1FAE5;
            color: #059669;
            border: 1px solid #6EE7B7;
        }
        
        /* Info alert boxes */
        .callout-box {
            background-color: #F8FAFC;
            border-left: 4px solid #2563EB;
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1rem 0;
            font-size: 0.92rem;
            color: #334155;
        }
        .callout-warning {
            border-left-color: #F59E0B;
            background-color: #FFFBEB;
        }
        
        /* Footer */
        .footer-text {
            font-size: 0.82rem;
            color: #94A3B8;
            text-align: center;
            padding: 2rem 0 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


def render_metric_card(title: str, value: str, subtitle: str = "", delta_color: str = None):
    """Renders an enterprise styled metric tile."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_filters() -> Tuple[float, float, str]:
    """Renders standardized threshold and model parameters in the sidebar."""
    st.sidebar.markdown("### ⚙️ Risk Configuration")
    high_thresh = st.sidebar.slider(
        "High Risk Cutoff (P >=)",
        min_value=0.50, max_value=0.85, value=0.60, step=0.05,
        help="Customers with predicted probability above this threshold trigger Critical/High priority retention alerts."
    )
    med_thresh = st.sidebar.slider(
        "Medium Risk Cutoff (P >=)",
        min_value=0.20, max_value=0.50, value=0.35, step=0.05,
        help="Customers between Medium and High cutoffs are flagged for early engagement monitoring."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B;">
            <b>Zelvora Industry Labs</b><br>
            AI Week 3: Level Up Project<br>
            Customer Churn Intelligence v1.0
        </div>
        """,
        unsafe_allow_html=True
    )

    return high_thresh, med_thresh


def render_footer():
    """Renders standard footer across pages."""
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-text">
            <b>Zelvora Technologies Pvt. Ltd. — Zelvora Industry Labs</b> | AI Week 3: Level Up Project<br>
            Customer Churn Prediction & Intelligent Retention Recommendation System | 
            Production Streamlit Dashboard
        </div>
        """,
        unsafe_allow_html=True
    )
