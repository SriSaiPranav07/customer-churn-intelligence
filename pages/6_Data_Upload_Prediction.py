"""Page 6: Batch Data Upload & Live Prediction.

CSV upload interface, schema validation, batch inference, and results export.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.data_validation import validate_uploaded_csv
from src.predict import get_predictor
from src.retention_recommender import generate_batch_recommendations
from src.ui_components import inject_custom_css, render_footer, render_sidebar_filters
from src.utils import DATA_DIR, format_currency

st.set_page_config(page_title="Batch Data Upload | Churn Intelligence", page_icon="📂", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("📂 Batch Customer Upload & Live Prediction")
st.markdown("Upload external customer CSV files for automated validation, batch churn scoring, and retention strategy generation.")

# Download sample template & Quick Demo Loader
sample_file_path = DATA_DIR / "sample_customers.csv"
col_t1, col_t2 = st.columns([1, 1])

with col_t1:
    if sample_file_path.exists():
        with open(sample_file_path, "rb") as f:
            sample_bytes = f.read()
        st.download_button(
            label="📄 Download Sample CSV Template (100 Customers)",
            data=sample_bytes,
            file_name="sample_customers_template.csv",
            mime="text/csv",
            use_container_width=True
        )

with col_t2:
    if st.button("🚀 Load Demo Customer Data (100 Accounts)", use_container_width=True):
        if sample_file_path.exists():
            st.session_state["demo_df"] = pd.read_csv(sample_file_path)
            st.session_state["demo_filename"] = "sample_customers.csv"

st.markdown("<br>", unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

raw_upload_df = None
file_source_name = ""

if uploaded_file is not None:
    try:
        raw_upload_df = pd.read_csv(uploaded_file)
        file_source_name = uploaded_file.name
        # Clear demo state when user explicitly uploads a file
        st.session_state.pop("demo_df", None)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
elif "demo_df" in st.session_state and st.session_state["demo_df"] is not None:
    raw_upload_df = st.session_state["demo_df"].copy()
    file_source_name = st.session_state.get("demo_filename", "sample_customers.csv")
    st.info("Loaded **Demo Sample Customer Data** (100 customer accounts). Ready for live scoring below.")

if raw_upload_df is not None:
    try:
        st.success(f"File ready: **{file_source_name}** ({len(raw_upload_df):,} rows, {len(raw_upload_df.columns)} columns).")

        # Preview uploaded data
        with st.expander("👁️ Preview Raw Upload Data (Top 5 Rows)", expanded=False):
            st.dataframe(raw_upload_df.head(5), use_container_width=True)

        is_pre_scored = "churn_probability" in raw_upload_df.columns

        if is_pre_scored:
            st.info("ℹ️ **Pre-Scored Batch Dataset Detected:** This file already contains churn risk probabilities (`churn_probability`). Analyzing portfolio risk and generating retention recommendations:")

            scored_df = raw_upload_df.copy()
            # Dynamically categorize using active sidebar threshold sliders
            scored_df["risk_category"] = [
                "High Risk" if p >= high_thresh else ("Medium Risk" if p >= med_thresh else "Low Risk")
                for p in scored_df["churn_probability"]
            ]

            # If retention recommendations are not present, generate them
            if "expected_retention_value" not in scored_df.columns or "primary_strategy" not in scored_df.columns:
                with st.spinner("Generating retention strategy recommendations..."):
                    recs_df = generate_batch_recommendations(scored_df)
            else:
                recs_df = scored_df

            # Summary metrics of the pre-scored batch
            high_count = (scored_df["risk_category"] == "High Risk").sum()
            med_count = (scored_df["risk_category"] == "Medium Risk").sum()
            low_count = (scored_df["risk_category"] == "Low Risk").sum()
            avg_p = float(scored_df["churn_probability"].mean())

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Total Accounts", f"{len(scored_df):,}")
            with b2:
                st.metric("High Risk Accounts", f"{high_count:,}", f"{high_count/len(scored_df)*100:.1f}%")
            with b3:
                st.metric("Medium Risk Accounts", f"{med_count:,}", f"{med_count/len(scored_df)*100:.1f}%")
            with b4:
                st.metric("Avg Predicted Risk", f"{avg_p*100:.1f}%")

            # Result Table View
            st.markdown("### 📋 Portfolio Risk & Retention Action Plan")
            view_cols = [
                c for c in [
                    "customerID", "risk_category", "churn_probability", "priority",
                    "primary_strategy", "primary_action", "expected_retention_value",
                    "estimated_cost", "revenue_protected", "Contract", "MonthlyCharges", "tenure"
                ] if c in recs_df.columns
            ]
            if not view_cols:
                view_cols = recs_df.columns.tolist()

            display_res = recs_df[view_cols].copy()
            if "churn_probability" in display_res.columns:
                display_res["churn_probability"] = display_res["churn_probability"].apply(lambda p: f"{float(p)*100:.1f}%")
            if "expected_retention_value" in display_res.columns:
                display_res["expected_retention_value"] = display_res["expected_retention_value"].apply(lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00")
            if "estimated_cost" in display_res.columns:
                display_res["estimated_cost"] = display_res["estimated_cost"].apply(lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00")
            if "revenue_protected" in display_res.columns:
                display_res["revenue_protected"] = display_res["revenue_protected"].apply(lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00")
            if "MonthlyCharges" in display_res.columns:
                display_res["MonthlyCharges"] = display_res["MonthlyCharges"].apply(lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00")

            st.dataframe(display_res, use_container_width=True, height=450, hide_index=True)

            # Export Button
            csv_out = recs_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Export Portfolio Risk & Retention Actions (CSV)",
                data=csv_out,
                file_name=f"updated_{file_source_name}",
                mime="text/csv",
                type="primary"
            )

        else:
            # Validation for raw un-scored customer records
            is_valid, errors, warnings, cleaned_df = validate_uploaded_csv(raw_upload_df)

            if warnings:
                with st.expander("⚠️ Data Processing Warnings", expanded=True):
                    for w in warnings:
                        st.warning(w)

            if not is_valid:
                st.error("Validation Failed! Please resolve the following errors before running inference:")
                for err in errors:
                    st.error(f"• {err}")
                st.stop()

            # Run Prediction
            st.markdown("---")
            if st.button("⚡ Run Live Churn Scoring & Retention Recommender", type="primary", use_container_width=True):
                with st.spinner("Executing feature engineering and model inference..."):
                    predictor = get_predictor()
                    if not predictor.is_ready():
                        st.error("⚠️ **Model artifacts are not loaded.** Please run `python -m src.train` to generate production model artifacts.")
                        st.stop()
                    scored_df = predictor.predict(cleaned_df, high_threshold=high_thresh, med_threshold=med_thresh)
                    recs_df = generate_batch_recommendations(scored_df)

                st.success(f"Batch prediction completed for **{len(scored_df):,} customer accounts**!")

                # Summary metrics of the uploaded batch
                high_count = (scored_df["risk_category"] == "High Risk").sum()
                med_count = (scored_df["risk_category"] == "Medium Risk").sum()
                low_count = (scored_df["risk_category"] == "Low Risk").sum()
                avg_p = float(scored_df["churn_probability"].mean())

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.metric("Total Accounts", f"{len(scored_df):,}")
                with b2:
                    st.metric("High Risk Accounts", f"{high_count:,}", f"{high_count/len(scored_df)*100:.1f}%")
                with b3:
                    st.metric("Medium Risk Accounts", f"{med_count:,}", f"{med_count/len(scored_df)*100:.1f}%")
                with b4:
                    st.metric("Avg Predicted Risk", f"{avg_p*100:.1f}%")

                # Result Table View
                st.markdown("### 📋 Prediction Results & Recommended Actions")
                view_cols = [
                    c for c in [
                        "customerID", "risk_category", "churn_probability", "priority",
                        "primary_strategy", "primary_action", "expected_retention_value",
                        "estimated_cost", "revenue_protected", "Contract", "MonthlyCharges", "tenure"
                    ] if c in recs_df.columns
                ]
                if not view_cols:
                    view_cols = recs_df.columns.tolist()

                display_res = recs_df[view_cols].copy()
                display_res["churn_probability"] = display_res["churn_probability"].apply(lambda p: f"{p*100:.1f}%")
                display_res["expected_retention_value"] = display_res["expected_retention_value"].apply(format_currency)
                display_res["estimated_cost"] = display_res["estimated_cost"].apply(format_currency)
                display_res["revenue_protected"] = display_res["revenue_protected"].apply(format_currency)
                display_res["MonthlyCharges"] = display_res["MonthlyCharges"].apply(format_currency)

                st.dataframe(display_res, use_container_width=True, height=450, hide_index=True)

                # Export Button
                csv_out = recs_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Full Prediction Results & Retention Actions (CSV)",
                    data=csv_out,
                    file_name=f"batch_predictions_{file_source_name}",
                    mime="text/csv",
                    type="primary"
                )

    except Exception as e:
        st.error(f"Error parsing CSV file: {e}")

else:
    st.info("Upload a CSV file or download the sample template above to test batch prediction.")

render_footer()
