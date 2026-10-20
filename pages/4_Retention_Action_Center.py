"""Page 4: Retention Action Center (Mentor Challenge).

Prioritized customer intervention queue, status tracking, and exportable retention action plan.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.retention_recommender import generate_batch_recommendations, RETENTION_DISCLAIMER
from src.utils import format_currency, format_percentage, ACTION_STATUS_OPTIONS
from src.ui_components import inject_custom_css, render_metric_card, render_sidebar_filters, render_footer

st.set_page_config(page_title="Retention Action Center | Churn Intelligence", page_icon="🎯", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("🎯 Retention Action Center")
st.markdown("Prioritized operational intervention queue for customer success and retention specialists.")

# Session state initialization
if "customer_actions" not in st.session_state:
    st.session_state.customer_actions = {}

# Load and batch recommend
@st.cache_data(show_spinner="Generating prioritized retention queue...")
def get_action_queue():
    df, _ = load_telco_data()
    predictor = get_predictor()
    scored = predictor.predict(df, high_threshold=0.60, med_threshold=0.35)
    recs = generate_batch_recommendations(scored)
    return recs

action_df = get_action_queue().copy()

# Dynamically apply current user sidebar thresholds
action_df["risk_category"] = [
    "High Risk" if p >= high_thresh else ("Medium Risk" if p >= med_thresh else "Low Risk")
    for p in action_df["churn_probability"]
]

# Update status from session state if previously modified
for i, row in action_df.iterrows():
    cid = row["customerID"]
    if cid in st.session_state.customer_actions:
        action_df.loc[i, "action_status"] = st.session_state.customer_actions[cid]

# Top KPI metrics
total_in_queue = len(action_df)
critical_count = (action_df["priority"] == "Critical").sum()
high_count = (action_df["priority"] == "High").sum()
total_budget = action_df["estimated_cost"].sum()
total_rev_protected = action_df["revenue_protected"].sum()

m1, m2, m3, m4 = st.columns(4)
with m1:
    render_metric_card("Total Accounts in Queue", f"{total_in_queue:,}", "Active retention roster")
with m2:
    render_metric_card("Critical & High Priority", f"{critical_count + high_count:,}", f"Critical: {critical_count:,} | High: {high_count:,}")
with m3:
    render_metric_card("Est. Intervention Budget", format_currency(total_budget), "Total cost to execute all plans")
with m4:
    render_metric_card("Protected Annual ARR", format_currency(total_rev_protected), f"Projected ROI: {total_rev_protected / max(total_budget, 1):.1f}x")

st.markdown("<br>", unsafe_allow_html=True)

# Filters for Queue
st.markdown("### 🔍 Queue Management & Filtering")
q_col1, q_col2, q_col3 = st.columns(3)

with q_col1:
    priority_filter = st.multiselect(
        "Filter by Priority",
        ["Critical", "High", "Medium", "Low"],
        default=["Critical", "High"]
    )

with q_col2:
    status_filter = st.multiselect(
        "Filter by Action Status",
        ACTION_STATUS_OPTIONS,
        default=["Planned", "Contacted", "In Progress"]
    )

with q_col3:
    search_cust = st.text_input("Filter by Customer ID", placeholder="Search ID...").strip()

filtered_queue = action_df.copy()
if priority_filter:
    filtered_queue = filtered_queue[filtered_queue["priority"].isin(priority_filter)]
if status_filter:
    filtered_queue = filtered_queue[filtered_queue["action_status"].isin(status_filter)]
if search_cust:
    filtered_queue = filtered_queue[filtered_queue["customerID"].str.contains(search_cust, case=False, na=False)]

# Sort by Churn Probability descending by default
filtered_queue = filtered_queue.sort_values("churn_probability", ascending=False)

st.markdown(f"**Showing {len(filtered_queue):,} actionable cases** in current retention queue.")

# Table View
display_queue = filtered_queue[[
    "customerID", "priority", "churn_probability", "action_status",
    "primary_strategy", "primary_action", "rationale",
    "estimated_cost", "revenue_protected", "Contract", "MonthlyCharges"
]].copy()

display_queue["churn_probability"] = display_queue["churn_probability"].apply(lambda p: f"{p*100:.1f}%")
display_queue["estimated_cost"] = display_queue["estimated_cost"].apply(format_currency)
display_queue["revenue_protected"] = display_queue["revenue_protected"].apply(format_currency)
display_queue["MonthlyCharges"] = display_queue["MonthlyCharges"].apply(format_currency)

st.dataframe(
    display_queue,
    use_container_width=True,
    height=450,
    hide_index=True
)

# Export Section
st.markdown("### 📤 Export Action Plan")
st.markdown("Download this prioritized intervention list to dispatch to HubSpot, Salesforce, or CRM outreach workflows.")

csv_export = filtered_queue.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Export Prioritized Retention Action Plan (CSV)",
    data=csv_export,
    file_name="retention_action_plan.csv",
    mime="text/csv",
    type="primary"
)

st.markdown(f"""
    <div class="callout-box" style="margin-top: 1.5rem;">
        <b>Decision-Support Guardrail:</b><br>
        {RETENTION_DISCLAIMER}
    </div>
""", unsafe_allow_html=True)

render_footer()
