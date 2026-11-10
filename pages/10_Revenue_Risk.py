"""Page 10: Revenue-at-Risk Analytics.

Transparent business exposure calculations: MRR at risk, annualized exposure,
customer value distribution, and illustrative intervention ROI analysis.

All estimates are clearly labelled and based on model predictions, not confirmed churn outcomes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.ui_components import inject_custom_css, render_footer, render_metric_card, render_sidebar_filters
from src.utils import (
    COLOR_HIGH_RISK,
    COLOR_LOW_RISK,
    COLOR_MEDIUM_RISK,
    DEFAULT_ACCEPTANCE_RATE,
    DEFAULT_ANNUAL_REVENUE_BASIS,
    DEFAULT_INTERVENTION_COST,
    format_currency,
)

st.set_page_config(
    page_title="Revenue-at-Risk Analytics | Churn Intelligence",
    page_icon="💰",
    layout="wide",
)
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("💰 Revenue-at-Risk Analytics")
st.markdown(
    "Quantifies estimated monthly and annualised recurring revenue exposure from predicted "
    "high-risk customer segments. All figures are **estimates** derived from model-predicted "
    "churn probabilities — not confirmed future outcomes."
)

st.info(
    "⚠️ **Disclaimer:** Revenue exposure estimates are based on predicted churn propensity "
    "scores from the IBM Telco cross-sectional dataset, not confirmed future churn events. "
    "Actual retention outcomes depend on intervention effectiveness, which should be validated "
    "via controlled experiments (A/B tests). Intervention cost and acceptance rate assumptions "
    "are illustrative defaults configurable in the sidebar."
)

# ---------------------------------------------------------------------------
# Load & Score Data (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading and scoring customer portfolio...")
def load_scored_data(_high_thresh: float, _med_thresh: float):
    """Score the full Telco dataset using the production champion model."""
    df, _meta = load_telco_data()
    predictor = get_predictor()
    if not predictor.is_ready():
        return None, "Model artifacts not found. Run `python -m src.train` first."
    scored = predictor.predict(df, high_threshold=_high_thresh, med_threshold=_med_thresh)
    return scored, None


scored_df, load_error = load_scored_data(high_thresh, med_thresh)

if load_error:
    st.error(load_error)
    st.stop()

if scored_df is None or scored_df.empty:
    st.error("No scored data available.")
    st.stop()

# Pull economic assumptions from session state (set by sidebar)
intervention_cost = st.session_state.get("intervention_cost", DEFAULT_INTERVENTION_COST)
acceptance_rate = st.session_state.get("acceptance_rate", DEFAULT_ACCEPTANCE_RATE)
annual_revenue_basis = st.session_state.get("annual_revenue", DEFAULT_ANNUAL_REVENUE_BASIS)

# Segment masks
mask_high = scored_df["risk_category"] == "High Risk"
mask_med = scored_df["risk_category"] == "Medium Risk"
mask_low = scored_df["risk_category"] == "Low Risk"

high_df = scored_df[mask_high]
med_df = scored_df[mask_med]
low_df = scored_df[mask_low]

# Monthly charges for each segment
high_mrr = float(high_df["MonthlyCharges"].sum()) if "MonthlyCharges" in high_df.columns else 0.0
med_mrr = float(med_df["MonthlyCharges"].sum()) if "MonthlyCharges" in med_df.columns else 0.0
total_mrr = float(scored_df["MonthlyCharges"].sum()) if "MonthlyCharges" in scored_df.columns else 0.0
annualised_exposure = high_mrr * 12
avg_churn_prob_high = float(high_df["churn_probability"].mean()) if not high_df.empty else 0.0

# ---------------------------------------------------------------------------
# Section 1: Executive KPI Cards
# ---------------------------------------------------------------------------
st.subheader("1. Executive Revenue Exposure Summary (Illustrative)")

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_metric_card("Total Customers", f"{len(scored_df):,}", "Full portfolio")
with k2:
    render_metric_card(
        "High-Risk Accounts",
        f"{len(high_df):,}",
        f"{len(high_df) / max(len(scored_df), 1) * 100:.1f}% of portfolio",
    )
with k3:
    render_metric_card(
        "MRR at Risk (High-Risk)",
        format_currency(high_mrr),
        "Monthly charges — high-risk segment",
    )
with k4:
    render_metric_card(
        "Annualised Exposure",
        format_currency(annualised_exposure),
        "High-risk MRR × 12 (illustrative)",
    )
with k5:
    render_metric_card(
        "Avg Churn Probability",
        f"{avg_churn_prob_high * 100:.1f}%",
        "High-risk segment mean",
    )

st.caption(
    "MRR at Risk = sum of MonthlyCharges for customers predicted as High Risk. "
    "These are estimates based on model predictions, not guaranteed revenue losses."
)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Risk Tier Distribution
# ---------------------------------------------------------------------------
st.subheader("2. Customer & Revenue Distribution by Risk Tier")

rc1, rc2 = st.columns(2)

tier_counts = pd.DataFrame({
    "Risk Tier": ["High Risk", "Medium Risk", "Low Risk"],
    "Customers": [len(high_df), len(med_df), len(low_df)],
    "MRR (USD)": [high_mrr, med_mrr, float(low_df["MonthlyCharges"].sum()) if "MonthlyCharges" in low_df.columns else 0],
})

with rc1:
    fig_pie = px.pie(
        tier_counts,
        names="Risk Tier",
        values="Customers",
        color="Risk Tier",
        color_discrete_map={
            "High Risk": COLOR_HIGH_RISK,
            "Medium Risk": COLOR_MEDIUM_RISK,
            "Low Risk": COLOR_LOW_RISK,
        },
        title="Customer Count by Risk Tier",
        hole=0.42,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label")
    fig_pie.update_layout(template="plotly_white", margin=dict(t=50, b=20), showlegend=False)
    st.plotly_chart(fig_pie, use_container_width=True)

with rc2:
    fig_mrr = px.bar(
        tier_counts,
        x="Risk Tier",
        y="MRR (USD)",
        color="Risk Tier",
        color_discrete_map={
            "High Risk": COLOR_HIGH_RISK,
            "Medium Risk": COLOR_MEDIUM_RISK,
            "Low Risk": COLOR_LOW_RISK,
        },
        text=tier_counts["MRR (USD)"].apply(format_currency),
        title="Monthly Recurring Revenue (MRR) by Risk Tier (Illustrative)",
    )
    fig_mrr.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(t=50, b=20),
        yaxis_title="Total Monthly Charges (USD)",
    )
    st.plotly_chart(fig_mrr, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3: Revenue Concentration (Lorenz-style)
# ---------------------------------------------------------------------------
st.subheader("3. High-Risk Revenue Concentration Analysis")

if "MonthlyCharges" in high_df.columns and not high_df.empty:
    sorted_charges = high_df["MonthlyCharges"].sort_values(ascending=False).reset_index(drop=True)
    cumulative_mrr = sorted_charges.cumsum()
    pct_customers = (sorted_charges.index + 1) / len(sorted_charges) * 100
    pct_mrr = cumulative_mrr / cumulative_mrr.iloc[-1] * 100

    fig_conc = go.Figure()
    fig_conc.add_trace(go.Scatter(
        x=pct_customers,
        y=pct_mrr,
        mode="lines",
        fill="tozeroy",
        name="MRR Concentration",
        line=dict(color=COLOR_HIGH_RISK, width=2.5),
        fillcolor="rgba(239,68,68,0.15)",
    ))
    # 80/20 reference lines
    fig_conc.add_hline(y=80, line_dash="dot", line_color="gray", annotation_text="80% MRR")
    fig_conc.add_vline(x=20, line_dash="dot", line_color="gray", annotation_text="Top 20%")

    fig_conc.update_layout(
        title="Cumulative MRR Concentration — High-Risk Segment",
        xaxis_title="Cumulative % of High-Risk Customers (sorted by Monthly Charges desc.)",
        yaxis_title="Cumulative % of High-Risk MRR",
        template="plotly_white",
        margin=dict(t=50, b=25),
        height=380,
    )
    st.plotly_chart(fig_conc, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Customer Value Distribution
# ---------------------------------------------------------------------------
st.subheader("4. Customer Value Distribution by Risk Category")

vd1, vd2 = st.columns(2)

with vd1:
    if "MonthlyCharges" in scored_df.columns:
        fig_hist = px.histogram(
            scored_df,
            x="MonthlyCharges",
            color="risk_category",
            color_discrete_map={
                "High Risk": COLOR_HIGH_RISK,
                "Medium Risk": COLOR_MEDIUM_RISK,
                "Low Risk": COLOR_LOW_RISK,
            },
            nbins=30,
            barmode="overlay",
            opacity=0.75,
            title="Monthly Charges Distribution by Risk Tier",
            labels={"MonthlyCharges": "Monthly Charges (USD)"},
        )
        fig_hist.update_layout(template="plotly_white", margin=dict(t=50, b=20), height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

with vd2:
    if "tenure" in scored_df.columns and "MonthlyCharges" in scored_df.columns:
        fig_scatter = px.scatter(
            scored_df.sample(min(1500, len(scored_df)), random_state=42),
            x="tenure",
            y="MonthlyCharges",
            color="risk_category",
            color_discrete_map={
                "High Risk": COLOR_HIGH_RISK,
                "Medium Risk": COLOR_MEDIUM_RISK,
                "Low Risk": COLOR_LOW_RISK,
            },
            opacity=0.55,
            size_max=6,
            title="Tenure vs Monthly Charges (coloured by Risk Tier)",
            labels={"tenure": "Customer Tenure (months)", "MonthlyCharges": "Monthly Charges (USD)"},
        )
        fig_scatter.update_layout(template="plotly_white", margin=dict(t=50, b=20), height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 5: Intervention ROI Analysis (Illustrative)
# ---------------------------------------------------------------------------
st.subheader("5. Retention Intervention Cost-Benefit Analysis (Illustrative)")

n_high = len(high_df)
total_intervention_cost = n_high * intervention_cost
expected_accepted = int(n_high * acceptance_rate)
expected_retained_value = expected_accepted * annual_revenue_basis
net_expected_value = expected_retained_value - total_intervention_cost

roi_col1, roi_col2 = st.columns([1.2, 1])
with roi_col1:
    thresholds = np.linspace(0.3, 0.9, 61)
    high_counts = [(scored_df["churn_probability"] >= t).sum() for t in thresholds]
    total_costs = [c * intervention_cost for c in high_counts]
    total_ev = [
        c * acceptance_rate * annual_revenue_basis - c * intervention_cost
        for c in high_counts
    ]

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=thresholds, y=total_costs,
        mode="lines", name="Total Outreach Cost",
        line=dict(color="#F59E0B", width=2.5),
    ))
    fig_roi.add_trace(go.Scatter(
        x=thresholds, y=total_ev,
        mode="lines", name="Net Expected Retention Value",
        line=dict(color="#10B981", width=2.5),
    ))
    fig_roi.add_vline(
        x=high_thresh, line_dash="dash", line_color="#EF4444",
        annotation_text=f"Current cutoff ({high_thresh:.2f})",
    )
    fig_roi.update_layout(
        title="Cost vs Expected Net Value by Threshold (Illustrative)",
        xaxis_title="High-Risk Cutoff Threshold",
        yaxis_title="USD (Illustrative)",
        template="plotly_white",
        margin=dict(t=50, b=25),
        height=380,
        hovermode="x unified",
    )
    st.plotly_chart(fig_roi, use_container_width=True)

with roi_col2:
    st.markdown("#### Intervention Assumptions")
    st.markdown(
        f"""
| Assumption | Value |
|---|---|
| High-risk customers targeted | **{n_high:,}** |
| Intervention cost per customer | **{format_currency(intervention_cost)}** |
| Estimated offer acceptance rate | **{acceptance_rate * 100:.0f}%** |
| Avg annual customer value basis | **{format_currency(annual_revenue_basis)}** |
| **Total outreach cost** | **{format_currency(total_intervention_cost)}** |
| Expected customers retained | **{expected_accepted:,}** |
| Estimated retained revenue | **{format_currency(expected_retained_value)}** |
| **Net expected value (illustrative)** | **{format_currency(net_expected_value)}** |
        """
    )

    if net_expected_value > 0:
        st.success(f"At current assumptions, expected net value is positive: **{format_currency(net_expected_value)}**")
    else:
        st.warning("At current assumptions, outreach cost exceeds expected retained value. Adjust thresholds or costs in the sidebar.")

    st.caption(
        "⚠️ These figures are illustrative estimates based on configurable assumptions. "
        "They do not represent guaranteed revenue outcomes. Validate with A/B testing."
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 6: At-Risk Customer Roster
# ---------------------------------------------------------------------------
st.subheader("6. High-Risk Customer Roster (Sortable)")

roster_cols = [
    c for c in [
        "customerID", "risk_category", "churn_probability",
        "MonthlyCharges", "TotalCharges", "tenure", "Contract",
        "InternetService", "PaymentMethod",
    ] if c in high_df.columns
]

roster_display = high_df[roster_cols].copy().sort_values("churn_probability", ascending=False)
if "churn_probability" in roster_display.columns:
    roster_display["churn_probability"] = roster_display["churn_probability"].apply(
        lambda p: f"{float(p) * 100:.1f}%"
    )
if "MonthlyCharges" in roster_display.columns:
    roster_display["MonthlyCharges"] = roster_display["MonthlyCharges"].apply(format_currency)
if "TotalCharges" in roster_display.columns:
    roster_display["TotalCharges"] = roster_display["TotalCharges"].apply(
        lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00"
    )

st.dataframe(roster_display, use_container_width=True, height=400, hide_index=True)

export_roster = high_df[roster_cols].copy().sort_values("churn_probability", ascending=False)
st.download_button(
    label=f"⬇️ Download High-Risk Roster ({n_high:,} accounts)",
    data=export_roster.to_csv(index=False).encode("utf-8"),
    file_name="high_risk_customer_roster.csv",
    mime="text/csv",
    type="primary",
)

render_footer()
