"""Page 3: Individual Customer Analysis.

Single-customer dossier, local SHAP explainability, and personalized retention action assignment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.explainability import get_shap_explainer, plot_customer_shap_waterfall
from src.retention_recommender import generate_customer_recommendation
from src.utils import (
    format_currency, format_percentage, get_risk_badge_color,
    ACTION_STATUS_OPTIONS, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
)
from src.ui_components import inject_custom_css, render_sidebar_filters, render_footer

st.set_page_config(page_title="Customer Analysis | Churn Intelligence", page_icon="👤", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("👤 Individual Customer Dossier & Explainability")
st.markdown("Detailed diagnostic of individual account health, underlying SHAP risk drivers, and customized retention workflows.")

# Initialize session state for customer action statuses
if "customer_actions" not in st.session_state:
    st.session_state.customer_actions = {}

# Load data and predictor
@st.cache_data(show_spinner="Loading customer data...")
def get_customer_pool():
    df, _ = load_telco_data()
    predictor = get_predictor()
    scored = predictor.predict(df, high_threshold=0.60, med_threshold=0.35)
    return scored

df_pool = get_customer_pool().copy()
df_pool["risk_category"] = [
    "High Risk" if p >= high_thresh else ("Medium Risk" if p >= med_thresh else "Low Risk")
    for p in df_pool["churn_probability"]
]

# Customer Selection
col_select1, col_select2 = st.columns([1.5, 1])

with col_select1:
    customer_ids = df_pool["customerID"].tolist()
    # Default to a high-risk customer for immediate demonstration value
    high_risk_ids = df_pool[df_pool["risk_category"] == "High Risk"]["customerID"].tolist()
    default_idx = customer_ids.index(high_risk_ids[0]) if high_risk_ids else 0

    selected_cust_id = st.selectbox(
        "Select Customer Account",
        customer_ids,
        index=default_idx,
        help="Select or type customer ID to inspect individual risk factors."
    )

customer_row = df_pool[df_pool["customerID"] == selected_cust_id].iloc[0]
churn_prob = float(customer_row["churn_probability"])
risk_category = str(customer_row["risk_category"])
badge_color = get_risk_badge_color(risk_category)

# Local SHAP Calculation
shap_explainer = get_shap_explainer()
with st.spinner("Computing local SHAP feature attributions..."):
    explanation = shap_explainer.explain_single_customer(customer_row)

# Profile and Risk Gauge Row
p_col1, p_col2 = st.columns([1.2, 1])

with p_col1:
    st.subheader("Account Profile")
    tenure_val = int(customer_row.get("tenure", 0))
    monthly_val = float(customer_row.get("MonthlyCharges", 0.0))
    total_val = float(customer_row.get("TotalCharges", 0.0))
    contract_val = str(customer_row.get("Contract", "Month-to-month"))
    pay_val = str(customer_row.get("PaymentMethod", "N/A"))
    net_val = str(customer_row.get("InternetService", "N/A"))

    st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.25rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #0F172A;">Customer ID: {selected_cust_id}</h3>
                <span style="background-color: {badge_color}20; color: {badge_color}; font-weight: 700; padding: 0.25rem 0.8rem; border-radius: 9999px; font-size: 0.85rem;">
                    {risk_category}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; font-size: 0.9rem;">
                <div><b>Tenure:</b> {tenure_val} months</div>
                <div><b>Monthly Fee:</b> {format_currency(monthly_val)}</div>
                <div><b>Total Billed:</b> {format_currency(total_val)}</div>
                <div><b>Contract:</b> {contract_val}</div>
                <div><b>Payment:</b> {pay_val}</div>
                <div><b>Internet:</b> {net_val}</div>
                <div><b>Tech Support:</b> {customer_row.get('TechSupport', 'No')}</div>
                <div><b>Online Security:</b> {customer_row.get('OnlineSecurity', 'No')}</div>
                <div><b>Paperless Billing:</b> {customer_row.get('PaperlessBilling', 'No')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with p_col2:
    st.subheader("Predicted Churn Probability")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_prob * 100,
        number=dict(suffix="%", font=dict(size=40, color="#0F172A")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor="#94A3B8"),
            bar=dict(color=badge_color, thickness=0.3),
            bgcolor="white",
            borderwidth=1,
            bordercolor="#E2E8F0",
            steps=[
                dict(range=[0, med_thresh * 100], color="#ECFDF5"),
                dict(range=[med_thresh * 100, high_thresh * 100], color="#FFFBEB"),
                dict(range=[high_thresh * 100, 100], color="#FEF2F2")
            ],
            threshold=dict(
                line=dict(color="#0F172A", width=3),
                thickness=0.75,
                value=churn_prob * 100
            )
        )
    ))
    fig_gauge.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=30, r=30),
        template="plotly_white"
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# SHAP Explainability Section
st.subheader("🔬 SHAP Explainable AI: Individual Churn Drivers")
st.markdown("Quantifying exactly which features pushed this customer's churn risk upward (red) or downward (green).")

shap_left, shap_right = st.columns([1.3, 1])

with shap_left:
    fig_shap = plot_customer_shap_waterfall(explanation, title=f"Feature Attributions for {selected_cust_id}")
    st.plotly_chart(fig_shap, use_container_width=True)

with shap_right:
    st.markdown("#### 📝 Executive Plain-English Diagnostic")
    st.info(explanation.get("narrative", "Diagnostic narrative unavailable."))

    st.markdown("#### ⚠️ Interpretation Guardrail")
    st.caption(explanation.get("disclaimer", ""))

st.markdown("<br>", unsafe_allow_html=True)

# Intelligent Retention Recommendation Section
st.subheader("💡 Intelligent Retention Action (Mentor Challenge)")

rec = generate_customer_recommendation(
    customer_row,
    churn_probability=churn_prob,
    shap_risk_drivers=explanation.get("risk_drivers", [])
)

# Fetch persisted action status if exists
current_status = st.session_state.customer_actions.get(selected_cust_id, rec["action_status"])

rec_box_left, rec_box_right = st.columns([1.6, 1])

with rec_box_left:
    st.markdown(f"""
        <div style="background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px; padding: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                    Recommended Strategy: {rec['primary_strategy']}
                </span>
                <span style="background-color: #EFF6FF; color: #1D4ED8; font-weight: 700; font-size: 0.8rem; padding: 0.25rem 0.7rem; border-radius: 6px;">
                    Priority: {rec['priority']}
                </span>
            </div>
            <p style="font-size: 1.05rem; font-weight: 600; color: #0F172A; margin-bottom: 0.75rem;">
                {rec['primary_action']}
            </p>
            <div style="background-color: #F8FAFC; border-left: 3px solid #2563EB; padding: 0.75rem 1rem; border-radius: 4px; font-size: 0.9rem; color: #475569; margin-bottom: 1rem;">
                <b>Business Rationale:</b> {rec['rationale']}
            </div>
            <div style="display: flex; gap: 2.5rem; font-size: 0.9rem;">
                <div>Estimated Intervention Cost: <b style="color: #0F172A;">{format_currency(rec['estimated_cost'])}</b></div>
                <div>Estimated Revenue Protected: <b style="color: #10B981;">{format_currency(rec['revenue_protected'])}</b></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with rec_box_right:
    st.markdown("#### 📋 Operational Action Status")
    new_status = st.selectbox(
        "Update Workflow Status",
        ACTION_STATUS_OPTIONS,
        index=ACTION_STATUS_OPTIONS.index(current_status) if current_status in ACTION_STATUS_OPTIONS else 0,
        key=f"status_select_{selected_cust_id}"
    )

    notes = st.text_area("Retention Representative Notes", placeholder="e.g. Called customer, discussed 1-year contract discount...", key=f"notes_{selected_cust_id}")

    if st.button("💾 Save Action Status", type="primary", use_container_width=True):
        st.session_state.customer_actions[selected_cust_id] = new_status
        st.success(f"Status updated to '{new_status}' for customer {selected_cust_id}!")

render_footer()
