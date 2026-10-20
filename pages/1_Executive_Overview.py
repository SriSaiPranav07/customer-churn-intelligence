"""Page 1: Executive Overview.

Macro customer churn analytics, financial exposure, and risk cohort breakdowns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.utils import (
    COLOR_HIGH_RISK, COLOR_MEDIUM_RISK, COLOR_LOW_RISK, COLOR_PRIMARY,
    format_currency, format_percentage, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
)
from src.ui_components import inject_custom_css, render_metric_card, render_sidebar_filters, render_footer

st.set_page_config(page_title="Executive Overview | Churn Intelligence", page_icon="📊", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("📊 Executive Overview & Portfolio Exposure")
st.markdown("Macro portfolio health, revenue-at-risk assessments, and cross-cohort churn vulnerability distributions.")

# Load data and run cached predictions
@st.cache_data(show_spinner="Evaluating portfolio risk...")
def get_portfolio_data():
    df, meta = load_telco_data()
    predictor = get_predictor()
    scored_df = predictor.predict(df, high_threshold=0.60, med_threshold=0.35)
    return scored_df, meta

scored_df, meta = get_portfolio_data()

# Re-categorize dynamically with current user sidebar thresholds
scored_df["risk_category"] = [
    "High Risk" if p >= high_thresh else ("Medium Risk" if p >= med_thresh else "Low Risk")
    for p in scored_df["churn_probability"]
]

# Calculations
total_customers = len(scored_df)
high_risk_df = scored_df[scored_df["risk_category"] == "High Risk"]
high_risk_count = len(high_risk_df)
avg_churn_prob = float(scored_df["churn_probability"].mean())
historical_churn_rate = float((scored_df["Churn"] == "Yes").mean()) if "Churn" in scored_df.columns else 0.2654

monthly_rev_at_risk = float(high_risk_df["MonthlyCharges"].sum())
annual_rev_at_risk = monthly_rev_at_risk * 12

# Top KPI row
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_metric_card("Analyzed Accounts", f"{total_customers:,}", "Total active customer records")
with c2:
    render_metric_card("High-Risk Accounts", f"{high_risk_count:,}", f"P ≥ {high_thresh:.2f} ({format_percentage(high_risk_count/total_customers)})")
with c3:
    render_metric_card("Observed Churn Rate", format_percentage(historical_churn_rate), "Historical ground truth baseline")
with c4:
    render_metric_card("Avg Churn Probability", format_percentage(avg_churn_prob), "Mean predicted probability")
with c5:
    render_metric_card("Monthly ARR Exposure", format_currency(monthly_rev_at_risk), f"Annual: {format_currency(annual_rev_at_risk)}")

st.markdown("<br>", unsafe_allow_html=True)

# Risk Distribution and Contract Breakdown
chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.subheader("Customer Risk Tier Distribution")
    risk_counts = scored_df["risk_category"].value_counts().reset_index()
    risk_counts.columns = ["Risk Tier", "Count"]

    color_map = {
        "High Risk": COLOR_HIGH_RISK,
        "Medium Risk": COLOR_MEDIUM_RISK,
        "Low Risk": COLOR_LOW_RISK
    }

    fig_pie = px.pie(
        risk_counts,
        values="Count",
        names="Risk Tier",
        hole=0.55,
        color="Risk Tier",
        color_discrete_map=color_map
    )
    fig_pie.update_traces(textinfo="percent+label", pull=[0.05, 0, 0])
    fig_pie.update_layout(template="plotly_white", margin=dict(t=20, b=20, l=10, r=10), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with chart_col2:
    st.subheader("Churn Probability by Contract Structure")
    contract_agg = scored_df.groupby("Contract")["churn_probability"].agg(["mean", "count"]).reset_index()
    contract_agg.columns = ["Contract", "Mean Churn Probability", "Customer Count"]

    fig_contract = px.bar(
        contract_agg,
        x="Contract",
        y="Mean Churn Probability",
        text=contract_agg["Mean Churn Probability"].apply(lambda v: f"{v*100:.1f}%"),
        color="Contract",
        color_discrete_sequence=["#EF4444", "#3B82F6", "#10B981"]
    )
    fig_contract.update_layout(
        template="plotly_white",
        yaxis_title="Mean Churn Probability",
        yaxis=dict(tickformat=".0%"),
        margin=dict(t=20, b=20, l=10, r=10),
        showlegend=False
    )
    st.plotly_chart(fig_contract, use_container_width=True)

# Tenure Cohort and Payment Method Analysis
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Churn Probability across Tenure Cohorts")
    scored_df["tenure_bracket"] = pd.cut(
        scored_df["tenure"],
        bins=[-1, 12, 24, 48, 60, 100],
        labels=["0-12m (New)", "13-24m (Early)", "25-48m (Mature)", "49-60m (Loyal)", ">60m (Veteran)"]
    )
    tenure_agg = scored_df.groupby("tenure_bracket", observed=False)["churn_probability"].mean().reset_index()

    fig_tenure = px.line(
        tenure_agg,
        x="tenure_bracket",
        y="churn_probability",
        markers=True,
        line_shape="spline",
        title=""
    )
    fig_tenure.update_traces(line=dict(color="#2563EB", width=3), marker=dict(size=9, color="#1D4ED8"))
    fig_tenure.update_layout(
        template="plotly_white",
        xaxis_title="Customer Tenure Lifecycle",
        yaxis_title="Average Predicted Churn Risk",
        yaxis=dict(tickformat=".0%"),
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_tenure, use_container_width=True)

with row2_col2:
    st.subheader("Financial Exposure by Payment Method")
    pay_agg = scored_df.groupby("PaymentMethod").agg(
        total_monthly_rev=("MonthlyCharges", "sum"),
        at_risk_monthly_rev=("MonthlyCharges", lambda s: s[scored_df.loc[s.index, "risk_category"] == "High Risk"].sum())
    ).reset_index()

    fig_pay = go.Figure()
    fig_pay.add_trace(go.Bar(
        name="Total Monthly Revenue",
        x=pay_agg["PaymentMethod"],
        y=pay_agg["total_monthly_rev"],
        marker_color="#94A3B8"
    ))
    fig_pay.add_trace(go.Bar(
        name="High-Risk Revenue",
        x=pay_agg["PaymentMethod"],
        y=pay_agg["at_risk_monthly_rev"],
        marker_color=COLOR_HIGH_RISK
    ))
    fig_pay.update_layout(
        barmode="group",
        template="plotly_white",
        xaxis_title="Payment Method",
        yaxis_title="Monthly Revenue ($)",
        margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(x=0.6, y=0.95)
    )
    st.plotly_chart(fig_pay, use_container_width=True)

# Methodology Disclosure Callout Box
st.markdown("""
    <div class="callout-box">
        <b>Methodological Clarification & Prediction Horizon:</b><br>
        Metrics displayed on this executive overview represent model-estimated churn probabilities calibrated on historical customer account behaviors. Because the benchmark IBM Telco dataset provides a static cross-sectional churn label rather than timestamped rolling events, predictions represent estimated churn risk rather than guaranteed future 30-day cancellations.
    </div>
""", unsafe_allow_html=True)

render_footer()
