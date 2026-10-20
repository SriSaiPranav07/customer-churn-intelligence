"""Customer Churn Prediction & Intelligent Retention Recommendation System.

Main Application Entrypoint & Executive Command Center.
Zelvora Technologies Pvt. Ltd. — Zelvora Industry Labs (AI Week 3 Level Up Project).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import (
    MODELS_DIR, DATA_DIR, get_risk_badge_color,
    format_currency, format_percentage, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
)
from src.ui_components import inject_custom_css, render_metric_card, render_sidebar_filters, render_footer
from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.retention_recommender import generate_customer_recommendation

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence | Zelvora Labs",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# 2. Data & Model Loading with Caching
@st.cache_data(show_spinner=False)
def load_app_data():
    df, meta = load_telco_data()
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return df, meta, metrics


@st.cache_resource(show_spinner=False)
def load_app_predictor():
    return get_predictor()


df_raw, dataset_meta, eval_metrics = load_app_data()
predictor = load_app_predictor()

# 3. Sidebar Filters
high_thresh, med_thresh = render_sidebar_filters()

# 4. Header & Overview
st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span style="background-color: #DBEAFE; color: #1D4ED8; font-size: 0.8rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em;">
            ZELVORA INDUSTRY LABS • WEEK 3 LEVEL UP
        </span>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #0F172A; margin-top: 0.5rem; margin-bottom: 0.4rem;">
            Customer Churn Intelligence & Retention System
        </h1>
        <p style="font-size: 1.05rem; color: #475569; max-width: 950px; line-height: 1.5;">
            An enterprise AI decision-support platform for telecom and SaaS subscription services.
            Predict churn probability, uncover underlying behavioral drivers with explainable AI (SHAP),
            and recommend high-ROI retention interventions before revenue is lost.
        </p>
    </div>
""", unsafe_allow_html=True)

# System Readiness Banner
champion_name = eval_metrics.get("metadata", {}).get("champion_model_name", "Random Forest")
test_auc = eval_metrics.get("models", {}).get(champion_name, {}).get("default_metrics", {}).get("roc_auc", 0.84)
is_synthetic = dataset_meta.get("is_synthetic", False)

st.markdown(f"""
    <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.75rem 1.2rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.8rem;">
        <div style="font-size: 0.9rem; color: #334155;">
            🟢 <b>Production Model:</b> {champion_name} &nbsp;|&nbsp; 
            🎯 <b>Held-Out Test ROC-AUC:</b> {test_auc:.4f} &nbsp;|&nbsp; 
            📁 <b>Dataset:</b> {dataset_meta.get('source')} ({len(df_raw):,} accounts)
        </div>
        <div style="font-size: 0.82rem; color: #64748B;">
            {'⚠️ Synthetic Demo Data' if is_synthetic else '✔️ Verified Benchmark'}
        </div>
    </div>
""", unsafe_allow_html=True)

# 5. Core KPI Cards
col1, col2, col3, col4 = st.columns(4)

total_cust = len(df_raw)
observed_churn_rate = (df_raw["Churn"] == "Yes").mean() if "Churn" in df_raw.columns else 0.265
total_monthly_revenue = df_raw["MonthlyCharges"].sum()

# Quick sample prediction for top KPIs
sample_preds = predictor.predict(df_raw.head(1000), high_threshold=high_thresh, med_threshold=med_thresh)
pct_high_risk = (sample_preds["risk_category"] == "High Risk").mean()
high_risk_revenue = (sample_preds[sample_preds["risk_category"] == "High Risk"]["MonthlyCharges"].sum() / len(sample_preds)) * total_cust

with col1:
    render_metric_card("Total Accounts Analyzed", f"{total_cust:,}", "Enterprise subscription base")
with col2:
    render_metric_card("Historical Churn Rate", format_percentage(observed_churn_rate), "Benchmark cohort churn rate")
with col3:
    render_metric_card("Predicted High Risk", format_percentage(pct_high_risk), f"Threshold P ≥ {high_thresh:.2f}")
with col4:
    render_metric_card("Monthly Revenue At Risk", format_currency(high_risk_revenue), "Estimated monthly at-risk ARR / 12")

st.markdown("<br>", unsafe_allow_html=True)

# 6. Interactive Customer Risk Simulator
st.markdown("### ⚡ Live Customer Churn Simulator")
st.markdown("Test instant churn probability prediction and tailored retention actions for any hypothetical or prospective customer profile.")

sim_col1, sim_col2, sim_col3 = st.columns(3)

with sim_col1:
    sim_tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=4)
    sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
    sim_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=1.0)

with sim_col2:
    sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
    sim_tech = st.selectbox("Tech Support Subscribed", ["No", "Yes"], index=0)
    sim_payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ], index=0)

with sim_col3:
    sim_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
    sim_partner = st.selectbox("Has Partner", ["Yes", "No"], index=1)
    sim_dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=1)

sim_total = round(sim_tenure * sim_monthly, 2)

sim_customer_data = pd.DataFrame([{
    "customerID": "SIM-CUSTOMER-01",
    "gender": "Male",
    "SeniorCitizen": sim_senior,
    "Partner": sim_partner,
    "Dependents": sim_dependents,
    "tenure": sim_tenure,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": sim_internet,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": sim_tech,
    "StreamingTV": "Yes" if sim_internet != "No" else "No internet service",
    "StreamingMovies": "Yes" if sim_internet != "No" else "No internet service",
    "Contract": sim_contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": sim_payment,
    "MonthlyCharges": sim_monthly,
    "TotalCharges": sim_total
}])

if st.button("🚀 Calculate Churn Risk & Retention Plan", type="primary", use_container_width=True):
    sim_res = predictor.predict(sim_customer_data, high_threshold=high_thresh, med_threshold=med_thresh)
    prob = float(sim_res.loc[0, "churn_probability"])
    cat = str(sim_res.loc[0, "risk_category"])
    badge_color = get_risk_badge_color(cat)

    rec = generate_customer_recommendation(sim_customer_data.iloc[0], prob)

    st.markdown("<br>", unsafe_allow_html=True)
    res_left, res_right = st.columns([1, 1.4])

    with res_left:
        st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 600;">PREDICTED CHURN RISK</div>
                <div style="font-size: 3.2rem; font-weight: 800; color: {badge_color}; margin: 0.5rem 0;">
                    {prob * 100:.1f}%
                </div>
                <div style="display: inline-block; background-color: {badge_color}20; color: {badge_color}; font-weight: 700; padding: 0.35rem 0.9rem; border-radius: 9999px; font-size: 0.95rem;">
                    {cat}
                </div>
                <div style="margin-top: 1rem; font-size: 0.85rem; color: #64748B;">
                    Simulated Tenure: <b>{sim_tenure} mos</b> | Monthly: <b>${sim_monthly}</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with res_right:
        st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span style="font-size: 0.85rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                        Intelligent Retention Action
                    </span>
                    <span style="background-color: #EFF6FF; color: #1D4ED8; font-size: 0.75rem; font-weight: 700; padding: 0.2rem 0.6rem; border-radius: 4px;">
                        Priority: {rec['priority']}
                    </span>
                </div>
                <h4 style="margin: 0 0 0.5rem 0; color: #0F172A;">{rec['primary_strategy']}</h4>
                <p style="font-size: 0.95rem; color: #334155; margin-bottom: 0.8rem;">
                    <b>Recommended Action:</b> {rec['primary_action']}
                </p>
                <div style="background-color: #F8FAFC; padding: 0.75rem; border-radius: 6px; font-size: 0.85rem; color: #475569; margin-bottom: 0.8rem;">
                    <b>Rationale:</b> {rec['rationale']}
                </div>
                <div style="display: flex; gap: 1.5rem; font-size: 0.85rem; color: #64748B;">
                    <div>Est. Cost: <b style="color: #0F172A;">{format_currency(rec['estimated_cost'])}</b></div>
                    <div>Est. Revenue Protected: <b style="color: #10B981;">{format_currency(rec['revenue_protected'])}</b></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<br><hr>", unsafe_allow_html=True)

# 7. Navigation Hub Cards
st.markdown("### 🧭 Platform Modules")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
        <div class="metric-card">
            <h4>📊 1. Executive Overview</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Macro KPI dashboard, revenue at risk, churn breakdown across contract types, payment channels, and tenure cohorts.
            </p>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>🔍 2. Customer Risk Explorer</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Filterable portfolio roster with customer search, risk tier classification, probability sorting, and CSV export.
            </p>
        </div>
    """, unsafe_allow_html=True)

with nav_col2:
    st.markdown("""
        <div class="metric-card">
            <h4>👤 3. Customer Analysis</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Deep-dive customer dossier with SHAP waterfall explanations, individual churn drivers, and action tracking.
            </p>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>🎯 4. Retention Action Center</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Prioritized queue for customer success teams to assign, execute, and monitor personalized interventions.
            </p>
        </div>
    """, unsafe_allow_html=True)

with nav_col3:
    st.markdown("""
        <div class="metric-card">
            <h4>📈 5. Model Performance</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Comparative evaluation across Logistic Regression, Random Forest, and XGBoost with ROC, PR, and threshold simulators.
            </p>
        </div>
        <div class="metric-card" style="margin-top: 1rem;">
            <h4>📂 6. Data Upload & Prediction</h4>
            <p style="font-size: 0.88rem; color: #64748B;">
                Batch inference pipeline with strict schema validation, instant probability scoring, and results download.
            </p>
        </div>
    """, unsafe_allow_html=True)

render_footer()
