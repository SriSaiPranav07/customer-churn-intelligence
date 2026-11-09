"""Shared UI components, styling injection, and layout utilities for Streamlit."""
from typing import Tuple

import streamlit as st

from src.utils import (
    DEFAULT_ACCEPTANCE_RATE,
    DEFAULT_ANNUAL_REVENUE_BASIS,
    DEFAULT_INTERVENTION_COST,
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


def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Renders an enterprise styled metric tile."""
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {f'<div class="metric-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
    """, unsafe_allow_html=True)


def render_sidebar_filters() -> Tuple[float, float]:
    """Renders standardized threshold, validation, and retention assumptions in the sidebar."""
    st.sidebar.markdown("### ⚙️ Risk Configuration")
    high_thresh = st.sidebar.slider(
        "High Risk Cutoff (P ≥)",
        min_value=0.45, max_value=0.85, value=0.60, step=0.05,
        help="Customers with predicted probability above this threshold trigger Critical/High priority retention alerts."
    )
    med_thresh = st.sidebar.slider(
        "Medium Risk Cutoff (P ≥)",
        min_value=0.15, max_value=0.55, value=0.35, step=0.05,
        help="Customers between Medium and High cutoffs are flagged for early engagement monitoring."
    )

    # Threshold order validation
    if med_thresh >= high_thresh:
        st.sidebar.error("⚠️ Validation Error: Medium Risk Cutoff must be strictly below High Risk Cutoff.")

    with st.sidebar.expander("💼 Retention Economics (Illustrative)", expanded=False):
        intervention_cost = st.number_input(
            "Intervention Cost ($)",
            min_value=1.0, max_value=200.0, value=float(DEFAULT_INTERVENTION_COST), step=5.0,
            help="Illustrative direct cost to execute an outreach intervention or discount."
        )
        acceptance_rate = st.slider(
            "Offer Acceptance Rate",
            min_value=0.05, max_value=1.0, value=float(DEFAULT_ACCEPTANCE_RATE), step=0.05,
            help="Illustrative percentage of contacted customers who accept retention offer."
        )
        annual_revenue = st.number_input(
            "Avg Annual Customer Value ($)",
            min_value=100.0, max_value=3000.0, value=float(DEFAULT_ANNUAL_REVENUE_BASIS), step=50.0,
            help="Illustrative annual subscription revenue baseline."
        )

        st.session_state["intervention_cost"] = intervention_cost
        st.session_state["acceptance_rate"] = acceptance_rate
        st.session_state["annual_revenue"] = annual_revenue

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B;">
            <b>Customer Churn Intelligence</b><br>
            Production Decision-Support System v1.1
        </div>
        """,
        unsafe_allow_html=True
    )

    return high_thresh, med_thresh


def render_footer():
    """Renders standard footer across pages with neutral industry wording."""
    st.markdown("---")
    st.markdown(
        """
        <div class="footer-text">
            <b>Customer Churn Intelligence & Retention Platform</b> | Production Decision-Support System<br>
            Empowering customer success and retention teams with explainable AI and proactive decision intelligence.
        </div>
        """,
        unsafe_allow_html=True
    )
