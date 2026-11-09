"""Customer Churn Prediction & Intelligent Retention Recommendation System.

Main Application Entrypoint & Executive Command Center.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import load_telco_data
from src.explainability import get_shap_explainer
from src.predict import get_predictor
from src.retention_recommender import generate_customer_recommendation
from src.ui_components import inject_custom_css, render_footer, render_metric_card, render_sidebar_filters
from src.utils import MODELS_DIR, format_currency, format_percentage, get_risk_badge_color

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Churn Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()

# 2. Data & Model Loading with Caching
@st.cache_data(show_spinner="Loading customer data...")
def load_app_data():
    df, meta = load_telco_data()
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            st.error(f"Failed to load evaluation metrics: {e}")
    else:
        st.error("Model artifacts and evaluation metrics are missing. Please run `python -m src.train` in your terminal.")
    return df, meta, metrics


@st.cache_resource(show_spinner="Initializing AI Model...")
def load_app_predictor():
    predictor = get_predictor()
    if not predictor.is_ready():
        predictor.load()
    return predictor


df_raw, dataset_meta, eval_metrics = load_app_data()
predictor = load_app_predictor()

if not predictor.is_ready():
    st.error("⚠️ **Model artifacts are not loaded or need retraining.** Run `python -m src.train` to generate production model artifacts.")
    st.stop()

# 3. Sidebar Filters
high_thresh, med_thresh = render_sidebar_filters()

# 4. Header & Overview
st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span style="background-color: #DBEAFE; color: #1D4ED8; font-size: 0.8rem; font-weight: 700; padding: 0.25rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em;">
            AI RETENTION ENGINEERING LAB
        </span>
        <h1 style="font-size: 2.3rem; font-weight: 800; color: #0F172A; margin-top: 0.5rem; margin-bottom: 0.4rem;">
            Customer Churn Intelligence & Retention System
        </h1>
        <p style="font-size: 1.05rem; color: #475569; max-width: 950px; line-height: 1.5;">
            An enterprise AI decision-support platform for telecommunications and subscription services.
            Predict churn probability, uncover underlying behavioral drivers with explainable AI (SHAP),
            and recommend high-ROI retention interventions before recurring revenue is lost.
        </p>
    </div>
""", unsafe_allow_html=True)

# Synthetic Data Warning
is_synthetic = dataset_meta.get("is_synthetic", False)
if is_synthetic:
    st.warning("⚠️ **Demonstration Notice:** Synthetic customer data is currently active. Predictions and metrics reflect simulated demonstration distributions.")

# System Readiness Banner (No Hardcoded Fallbacks)
if eval_metrics and "metadata" in eval_metrics:
    champion_name = eval_metrics.get("metadata", {}).get("champion_model_name", "n/a")
    champ_calib = eval_metrics.get("metadata", {}).get("champion_calibration", "Standard")
    test_auc_val = eval_metrics.get("models", {}).get(champion_name, {}).get("default_metrics", {}).get("roc_auc")
    test_auc_str = f"{test_auc_val:.4f}" if test_auc_val is not None else "n/a"

    st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 8px; padding: 0.75rem 1.2rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.8rem;">
            <div style="font-size: 0.9rem; color: #334155;">
                🟢 <b>Production Champion:</b> {champion_name} ({champ_calib}) &nbsp;|&nbsp;
                🎯 <b>Held-Out Test ROC-AUC:</b> {test_auc_str} &nbsp;|&nbsp;
                📁 <b>Dataset:</b> {dataset_meta.get('source')} ({len(df_raw):,} accounts)
            </div>
            <div style="font-size: 0.82rem; color: #64748B;">
                {'⚠️ Synthetic Demo' if is_synthetic else '✔️ Verified Benchmark'}
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.error("Model performance metrics not loaded. Run `python -m src.train` to generate model artifacts.")

# 5. Core KPI Cards (Using Full Out-Of-Fold Probabilities for All Rows)
@st.cache_data(show_spinner=False)
def compute_kpis_from_oof(high_th: float, med_th: float):
    cache_path = MODELS_DIR / "test_evaluation_cache.joblib"
    if cache_path.exists():
        try:
            cache = joblib.load(cache_path)
            oof_df = cache.get("oof_df")
            if oof_df is not None and len(oof_df) == len(df_raw):
                return oof_df
        except Exception:
            pass

    # Fallback to predictor if cache missing
    return predictor.predict(df_raw, high_threshold=high_th, med_threshold=med_th)

scored_portfolio = compute_kpis_from_oof(high_thresh, med_thresh)

total_cust = len(scored_portfolio)
if "Churn" in scored_portfolio.columns:
    observed_churn_rate = (scored_portfolio["Churn"] == "Yes").mean()
    obs_churn_str = format_percentage(observed_churn_rate)
    # Count MRR at risk strictly for active customers (Churn == "No")
    active_customers = scored_portfolio[scored_portfolio["Churn"] == "No"]
else:
    obs_churn_str = "n/a"
    active_customers = scored_portfolio

high_risk_all = scored_portfolio[scored_portfolio["churn_probability"] >= high_thresh]
pct_high_risk = len(high_risk_all) / total_cust if total_cust > 0 else 0.0

active_high_risk = active_customers[active_customers["churn_probability"] >= high_thresh]
mrr_at_risk = float(active_high_risk["MonthlyCharges"].sum())
arr_at_risk = mrr_at_risk * 12.0

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_metric_card("Total Accounts Analyzed", f"{total_cust:,}", "Full customer subscriber base")
with col2:
    render_metric_card("Historical Churn Rate", obs_churn_str, "Benchmark cohort ground truth")
with col3:
    render_metric_card("Predicted High Risk", format_percentage(pct_high_risk), f"Threshold P ≥ {high_thresh:.2f} ({len(high_risk_all):,} accounts)")
with col4:
    render_metric_card("MRR at Risk (Active)", format_currency(mrr_at_risk), f"Annualized ARR: {format_currency(arr_at_risk)}")

st.markdown("<br>", unsafe_allow_html=True)

# 6. Interactive Customer Risk Simulator
st.markdown("### ⚡ Live Customer Churn Simulator")
st.markdown("Test instant churn probability prediction, top SHAP drivers, and what-if retention actions for any hypothetical or prospective customer profile.")

# Use st.form so simulator inputs and outputs persist reliably across reruns
with st.form("churn_simulator_form"):
    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        sim_tenure = st.slider("Customer Tenure (Months)", min_value=1, max_value=72, value=4)
        sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
        sim_monthly = st.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=85.0, step=1.0)

    with sim_col2:
        sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
        # Realism guardrail: If Internet is "No", TechSupport is disabled and forced to "No internet service"
        sim_tech = st.selectbox(
            "Tech Support Subscribed",
            ["No", "Yes"] if sim_internet != "No" else ["No internet service"],
            disabled=(sim_internet == "No")
        )
        sim_payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], index=0)

    with sim_col3:
        sim_senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=0)
        sim_partner = st.selectbox("Has Partner", ["Yes", "No"], index=1)
        sim_dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=1)

    with st.expander("🛠️ Advanced Options & Demographic Inputs", expanded=False):
        adv_col1, adv_col2, adv_col3 = st.columns(3)
        with adv_col1:
            sim_gender = st.selectbox("Gender", ["Male", "Female"], index=0)
            sim_phone = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            sim_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"] if sim_phone == "Yes" else ["No phone service"])
        with adv_col2:
            sim_paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            sim_security = st.selectbox(
                "Online Security",
                ["No", "Yes"] if sim_internet != "No" else ["No internet service"],
                disabled=(sim_internet == "No")
            )
            sim_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes"] if sim_internet != "No" else ["No internet service"],
                disabled=(sim_internet == "No")
            )
        with adv_col3:
            sim_device = st.selectbox(
                "Device Protection",
                ["No", "Yes"] if sim_internet != "No" else ["No internet service"],
                disabled=(sim_internet == "No")
            )
            sim_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No"] if sim_internet != "No" else ["No internet service"],
                disabled=(sim_internet == "No")
            )
            sim_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No"] if sim_internet != "No" else ["No internet service"],
                disabled=(sim_internet == "No")
            )

    sim_submitted = st.form_submit_button("🚀 Calculate Churn Risk & Retention Plan", type="primary", use_container_width=True)

if sim_submitted:
    sim_total = round(sim_tenure * sim_monthly, 2)
    sim_customer_data = pd.DataFrame([{
        "customerID": "SIM-CUSTOMER-01",
        "gender": sim_gender,
        "SeniorCitizen": sim_senior,
        "Partner": sim_partner,
        "Dependents": sim_dependents,
        "tenure": sim_tenure,
        "PhoneService": sim_phone,
        "MultipleLines": sim_lines,
        "InternetService": sim_internet,
        "OnlineSecurity": "No internet service" if sim_internet == "No" else sim_security,
        "OnlineBackup": "No internet service" if sim_internet == "No" else sim_backup,
        "DeviceProtection": "No internet service" if sim_internet == "No" else sim_device,
        "TechSupport": "No internet service" if sim_internet == "No" else sim_tech,
        "StreamingTV": "No internet service" if sim_internet == "No" else sim_tv,
        "StreamingMovies": "No internet service" if sim_internet == "No" else sim_movies,
        "Contract": sim_contract,
        "PaperlessBilling": sim_paperless,
        "PaymentMethod": sim_payment,
        "MonthlyCharges": sim_monthly,
        "TotalCharges": sim_total
    }])

    try:
        sim_res = predictor.predict(sim_customer_data, high_threshold=high_thresh, med_threshold=med_thresh)
        prob = float(sim_res.loc[0, "churn_probability"])
        cat = str(sim_res.loc[0, "risk_category"])
        badge_color = get_risk_badge_color(cat)

        # Get SHAP drivers
        shap_explainer = get_shap_explainer()
        explanation = shap_explainer.explain_single_customer(sim_customer_data.iloc[0])
        risk_drivers = explanation.get("risk_drivers", [])

        # Generate recommendation
        rec = generate_customer_recommendation(sim_customer_data.iloc[0], prob, shap_risk_drivers=risk_drivers)

        # What-if Re-scoring: Apply recommended intervention to hypothetical profile
        what_if_data = sim_customer_data.copy()
        strat = rec["primary_strategy"]
        if "Contract" in strat:
            what_if_data["Contract"] = "One year"
        elif "AutoPay" in strat or "Billing" in strat:
            what_if_data["PaymentMethod"] = "Bank transfer (automatic)"
        elif "VIP" in strat or "Technical" in strat or "Value Realization" in strat:
            if what_if_data.loc[0, "InternetService"] != "No":
                what_if_data["TechSupport"] = "Yes"
                what_if_data["OnlineSecurity"] = "Yes"
        elif "Tier" in strat or "Loyalty" in strat:
            what_if_data["MonthlyCharges"] = round(what_if_data.loc[0, "MonthlyCharges"] * 0.90, 2)

        what_if_res = predictor.predict(what_if_data, high_threshold=high_thresh, med_threshold=med_thresh)
        new_prob = float(what_if_res.loc[0, "churn_probability"])
        delta_prob = new_prob - prob

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
                        Tenure: <b>{sim_tenure} mos</b> | Fee: <b>${sim_monthly:.2f}/mo</b>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Top 3 SHAP Drivers
            if risk_drivers:
                st.markdown("<div style='margin-top: 1rem;'><b>Top Risk Factors (SHAP):</b></div>", unsafe_allow_html=True)
                for d in risk_drivers[:3]:
                    st.markdown(f"• **{d['feature']}**: <span style='color:#EF4444;'>+{d['shap_value']:.2f}</span>", unsafe_allow_html=True)

        with res_right:
            st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <span style="font-size: 0.85rem; font-weight: 700; color: #2563EB; text-transform: uppercase;">
                            Targeted Retention Strategy
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
                        <b>Business Rationale:</b> {rec['rationale']}
                    </div>
                    <div style="display: flex; gap: 1.5rem; font-size: 0.85rem; color: #64748B; margin-bottom: 1rem;">
                        <div>Est. Cost: <b style="color: #0F172A;">{format_currency(rec['estimated_cost'])}</b></div>
                        <div>Est. Revenue Protected: <b style="color: #10B981;">{format_currency(rec['revenue_protected'])}</b></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # What-If Impact Box
            delta_color = "#10B981" if delta_prob <= 0 else "#EF4444"
            st.markdown(f"""
                <div style="background-color: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 8px; padding: 0.9rem; margin-top: 0.75rem;">
                    <div style="font-size: 0.85rem; font-weight: 700; color: #065F46;">
                        🔄 What-If Action Impact (Post-Intervention Rescore)
                    </div>
                    <div style="font-size: 1.1rem; color: #0F172A; margin-top: 0.25rem;">
                        Projected New Churn Risk: <b>{new_prob * 100:.1f}%</b>
                        (<span style="color: {delta_color}; font-weight: 700;">{delta_prob * 100:+.1f}%</span>)
                    </div>
                </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Simulation failed: {e}")

st.markdown("<br><hr>", unsafe_allow_html=True)

# 7. Navigation Hub (Using Modern st.page_link)
st.markdown("### 🧭 Platform Modules")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    with st.container(border=True):
        st.page_link("pages/1_Executive_Overview.py", label="1. Executive Overview", icon="📊")
        st.caption("Macro KPI dashboard, portfolio exposure, and churn distribution by contract and tenure.")
    with st.container(border=True):
        st.page_link("pages/2_Customer_Risk_Explorer.py", label="2. Customer Risk Explorer", icon="🔍")
        st.caption("Filterable portfolio roster with customer ID search, risk categorization, and CSV export.")

with nav_col2:
    with st.container(border=True):
        st.page_link("pages/3_Customer_Analysis.py", label="3. Individual Customer Dossier", icon="👤")
        st.caption("Single-customer diagnostics, local SHAP waterfall feature attributions, and workflow tracking.")
    with st.container(border=True):
        st.page_link("pages/4_Retention_Action_Center.py", label="4. Retention Action Center", icon="🎯")
        st.caption("Prioritized operational intervention queue, budget planning, and exportable action rosters.")

with nav_col3:
    with st.container(border=True):
        st.page_link("pages/5_Model_Performance.py", label="5. Model Performance & Evaluation", icon="📈")
        st.caption("Comparative metrics, ROC/PR curves, probability calibration, fairness audits, and threshold profit curve.")
    with st.container(border=True):
        st.page_link("pages/6_Data_Upload_Prediction.py", label="6. Batch Upload & Prediction", icon="📂")
        st.caption("Batch CSV scoring pipeline with strict schema validation and exportable retention plans.")

render_footer()
