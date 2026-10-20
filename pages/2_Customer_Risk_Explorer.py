"""Page 2: Customer Risk Explorer.

Portfolio roster exploration, multi-criteria filtering, and priority customer extraction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_telco_data
from src.predict import get_predictor
from src.retention_recommender import generate_customer_recommendation
from src.utils import format_currency, format_percentage
from src.ui_components import inject_custom_css, render_sidebar_filters, render_footer

st.set_page_config(page_title="Customer Risk Explorer | Churn Intelligence", page_icon="🔍", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("🔍 Customer Risk Explorer")
st.markdown("Search, filter, and inspect customer churn risk rankings across contract structures, tenure, and subscription tiers.")

# Load data and run predictions
@st.cache_data(show_spinner="Loading customer portfolio...")
def get_explorer_data():
    df, _ = load_telco_data()
    predictor = get_predictor()
    scored = predictor.predict(df, high_threshold=0.60, med_threshold=0.35)
    return scored

df_scored = get_explorer_data().copy()

# Dynamically apply current user sidebar thresholds
df_scored["risk_category"] = [
    "High Risk" if p >= high_thresh else ("Medium Risk" if p >= med_thresh else "Low Risk")
    for p in df_scored["churn_probability"]
]

# Filtering Controls
st.markdown("### 🛠️ Portfolio Filters")
f_col1, f_col2, f_col3, f_col4 = st.columns(4)

with f_col1:
    search_id = st.text_input("Search Customer ID", placeholder="e.g. 7590-VHVEG").strip()

with f_col2:
    selected_tier = st.selectbox("Risk Category", ["All", "High Risk", "Medium Risk", "Low Risk"], index=0)

with f_col3:
    selected_contract = st.selectbox("Contract Type", ["All"] + list(df_scored["Contract"].unique()), index=0)

with f_col4:
    selected_internet = st.selectbox("Internet Service", ["All"] + list(df_scored["InternetService"].unique()), index=0)

prob_range = st.slider(
    "Churn Probability Range",
    min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05
)

# Apply filters
filtered_df = df_scored.copy()

if search_id:
    filtered_df = filtered_df[filtered_df["customerID"].astype(str).str.contains(search_id, case=False, na=False)]

if selected_tier != "All":
    filtered_df = filtered_df[filtered_df["risk_category"] == selected_tier]

if selected_contract != "All":
    filtered_df = filtered_df[filtered_df["Contract"] == selected_contract]

if selected_internet != "All":
    filtered_df = filtered_df[filtered_df["InternetService"] == selected_internet]

filtered_df = filtered_df[
    (filtered_df["churn_probability"] >= prob_range[0]) &
    (filtered_df["churn_probability"] <= prob_range[1])
]

# Sorting
sort_order = st.radio("Sort Order", ["Highest Churn Risk First", "Lowest Churn Risk First"], horizontal=True)
if sort_order == "Highest Churn Risk First":
    filtered_df = filtered_df.sort_values("churn_probability", ascending=False)
else:
    filtered_df = filtered_df.sort_values("churn_probability", ascending=True)

# Display summary count
st.markdown(f"**Showing {len(filtered_df):,} matching customers** out of {len(df_scored):,} total accounts.")

# Prepare display table with key columns
display_cols = [
    "customerID", "risk_category", "churn_probability",
    "Contract", "tenure", "MonthlyCharges", "TotalCharges",
    "InternetService", "PaymentMethod"
]
table_df = filtered_df[display_cols].copy()
table_df["churn_probability"] = table_df["churn_probability"].apply(lambda p: f"{p*100:.1f}%")
table_df["MonthlyCharges"] = table_df["MonthlyCharges"].apply(format_currency)
table_df["TotalCharges"] = table_df["TotalCharges"].apply(format_currency)

st.dataframe(
    table_df,
    use_container_width=True,
    height=450,
    hide_index=True
)

# CSV Export
st.markdown("### 📥 Export Filtered Roster")
csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Filtered Customer List (CSV)",
    data=csv_data,
    file_name="filtered_customer_churn_risk.csv",
    mime="text/csv",
    type="primary"
)

render_footer()
