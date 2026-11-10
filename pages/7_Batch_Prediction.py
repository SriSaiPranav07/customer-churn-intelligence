"""Page 7: Batch Prediction & Portfolio Scoring Center.

CSV upload interface, schema validation, batch inference, progress tracking,
filtered roster downloads, and batch executive reports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import pandas as pd
import streamlit as st

from src.data_validation import validate_uploaded_csv
from src.predict import get_predictor
from src.retention_recommender import generate_batch_recommendations
from src.ui_components import inject_custom_css, render_footer, render_sidebar_filters
from src.utils import DATA_DIR, format_currency

st.set_page_config(
    page_title="Batch Prediction | Churn Intelligence",
    page_icon="📂",
    layout="wide",
)
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("📂 Batch Prediction & Portfolio Scoring Center")
st.markdown(
    "Upload external customer CSV files for automated validation, batch churn scoring, "
    "risk stratification, and targeted retention strategy dispatch."
)

# ---------------------------------------------------------------------------
# Template Download & Demo Loader
# ---------------------------------------------------------------------------
sample_file_path = DATA_DIR / "sample_customers.csv"
col_t1, col_t2 = st.columns(2)

with col_t1:
    if sample_file_path.exists():
        with open(sample_file_path, "rb") as _f:
            sample_bytes = _f.read()
        st.download_button(
            label="📄 Download Sample CSV Template (100 Customers)",
            data=sample_bytes,
            file_name="sample_customers_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.info("Sample template file not found in `data/` directory.")

with col_t2:
    if st.button("🚀 Load Demo Customer Data (100 Accounts)", use_container_width=True):
        if sample_file_path.exists():
            st.session_state["demo_df"] = pd.read_csv(sample_file_path)
            st.session_state["demo_filename"] = "sample_customers.csv"
        else:
            st.warning("Demo sample file not found.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# File Uploader
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

raw_upload_df = None
file_source_name = ""

if uploaded_file is not None:
    try:
        raw_upload_df = pd.read_csv(uploaded_file)
        file_source_name = uploaded_file.name
        st.session_state.pop("demo_df", None)
    except Exception as exc:
        st.error(f"Error reading CSV file: {exc}")
elif st.session_state.get("demo_df") is not None:
    raw_upload_df = st.session_state["demo_df"].copy()
    file_source_name = st.session_state.get("demo_filename", "sample_customers.csv")
    st.info("Loaded **Demo Sample Customer Data** (100 customer accounts). Ready for live scoring below.")

# ---------------------------------------------------------------------------
# Main processing block
# ---------------------------------------------------------------------------
if raw_upload_df is not None:
    try:
        st.success(
            f"File ready: **{file_source_name}** "
            f"({len(raw_upload_df):,} rows, {len(raw_upload_df.columns)} columns)."
        )

        with st.expander("👁️ Preview Raw Upload Data (Top 5 Rows)", expanded=False):
            st.dataframe(raw_upload_df.head(5), use_container_width=True)

        # ----------------------------------------------------------------
        # Route 1: Pre-scored output file detected – skip re-inference
        # ----------------------------------------------------------------
        is_pre_scored = "churn_probability" in raw_upload_df.columns

        if is_pre_scored:
            st.warning(
                "**Notice:** The uploaded file already contains churn predictions "
                "(`churn_probability`). This looks like a **previously exported output file**, "
                "not raw customer input data.\n\n"
                "The file contains **already-scored predictions and retention recommendations**, "
                "rather than the raw customer demographic and service attributes needed to "
                "generate new predictions.\n\n"
                "**How to test scoring:**\n"
                "1. Click the '🚀 Load Demo Customer Data' button above to test with 100 "
                "sample accounts instantly.\n"
                "2. Or click '📄 Download Sample CSV Template' to see the required raw feature columns."
            )

            scored_df = raw_upload_df.copy()
            # Re-band risk using current sidebar thresholds
            if "churn_probability" in scored_df.columns:
                scored_df["risk_category"] = scored_df["churn_probability"].apply(
                    lambda p: (
                        "High Risk" if float(p) >= high_thresh
                        else ("Medium Risk" if float(p) >= med_thresh else "Low Risk")
                    )
                )

            # Generate recommendations if not already present
            if "primary_strategy" not in scored_df.columns:
                with st.spinner("Generating retention strategy recommendations..."):
                    recs_df = generate_batch_recommendations(scored_df)
            else:
                recs_df = scored_df

            # KPI cards
            high_count = int((recs_df.get("risk_category", pd.Series()) == "High Risk").sum())
            med_count = int((recs_df.get("risk_category", pd.Series()) == "Medium Risk").sum())
            avg_p = float(scored_df["churn_probability"].astype(float).mean())

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Accounts", f"{len(scored_df):,}")
            b2.metric("High Risk", f"{high_count:,}", f"{high_count / max(len(scored_df), 1) * 100:.1f}%")
            b3.metric("Medium Risk", f"{med_count:,}", f"{med_count / max(len(scored_df), 1) * 100:.1f}%")
            b4.metric("Avg Predicted Risk", f"{avg_p * 100:.1f}%")

            # Result Table
            st.markdown("### 📋 Portfolio Risk & Retention Action Plan")
            view_cols = [
                c for c in [
                    "customerID", "risk_category", "churn_probability", "priority",
                    "primary_strategy", "primary_action", "expected_retention_value",
                    "estimated_cost", "revenue_protected", "Contract", "MonthlyCharges", "tenure",
                ] if c in recs_df.columns
            ]
            if not view_cols:
                view_cols = recs_df.columns.tolist()

            display_df = recs_df[view_cols].copy()
            if "churn_probability" in display_df:
                display_df["churn_probability"] = display_df["churn_probability"].apply(
                    lambda p: f"{float(p) * 100:.1f}%"
                )
            for money_col in ["expected_retention_value", "estimated_cost", "revenue_protected", "MonthlyCharges"]:
                if money_col in display_df:
                    display_df[money_col] = display_df[money_col].apply(
                        lambda v: format_currency(float(v)) if pd.notna(v) else "$0.00"
                    )

            st.dataframe(display_df, use_container_width=True, height=420, hide_index=True)

            # Downloads
            st.markdown("### 📥 Download Options")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    label="⬇️ Export Complete Batch (CSV)",
                    data=recs_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"batch_all_{file_source_name}",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )
            with ec2:
                high_risk_export = recs_df[recs_df.get("risk_category", pd.Series(dtype=str)) == "High Risk"]
                st.download_button(
                    label=f"⬇️ Export High-Risk Priority Roster ({high_count:,} accounts)",
                    data=high_risk_export.to_csv(index=False).encode("utf-8"),
                    file_name=f"high_risk_{file_source_name}",
                    mime="text/csv",
                    use_container_width=True,
                )

        # ----------------------------------------------------------------
        # Route 2: Raw un-scored customer records – validate then predict
        # ----------------------------------------------------------------
        else:
            is_valid, errors, warnings, cleaned_df = validate_uploaded_csv(raw_upload_df)

            if warnings:
                with st.expander("⚠️ Data Processing & Imputation Warnings", expanded=True):
                    for w in warnings:
                        st.warning(w)

            if not is_valid:
                st.error("**Validation Failed!** Please resolve the following errors before running inference:")
                for err in errors:
                    st.error(f"• {err}")
                st.info(
                    "**How to fix:** Download the sample CSV template above to see the required "
                    "columns. The required raw feature columns are listed in the template."
                )
                st.stop()

            # Run prediction on button click
            st.markdown("---")
            if st.button(
                "⚡ Run Live Churn Scoring & Retention Recommender",
                type="primary",
                use_container_width=True,
            ):
                predictor = get_predictor()
                if not predictor.is_ready():
                    st.error(
                        "⚠️ **Model artifacts are not loaded.** "
                        "Please run `python -m src.train` to generate production model artifacts."
                    )
                    st.stop()

                prog_bar = st.progress(0, text="Initializing inference pipeline...")
                prog_bar.progress(30, text="Executing feature engineering & transformations...")
                time.sleep(0.1)
                prog_bar.progress(60, text="Generating model churn probabilities...")
                scored_df = predictor.predict(cleaned_df, high_threshold=high_thresh, med_threshold=med_thresh)
                prog_bar.progress(85, text="Synthesizing personalized retention recommendations...")
                recs_df = generate_batch_recommendations(scored_df)
                prog_bar.progress(100, text="Batch prediction completed!")
                time.sleep(0.3)
                prog_bar.empty()

                st.success(f"✅ Batch prediction completed for **{len(scored_df):,} customer accounts**!")

                high_count = int((scored_df["risk_category"] == "High Risk").sum())
                med_count = int((scored_df["risk_category"] == "Medium Risk").sum())
                avg_p = float(scored_df["churn_probability"].mean())

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Accounts", f"{len(scored_df):,}")
                b2.metric("High Risk", f"{high_count:,}", f"{high_count / max(len(scored_df), 1) * 100:.1f}%")
                b3.metric("Medium Risk", f"{med_count:,}", f"{med_count / max(len(scored_df), 1) * 100:.1f}%")
                b4.metric("Avg Predicted Risk", f"{avg_p * 100:.1f}%")

                # Risk distribution chart
                import plotly.express as px
                risk_counts = scored_df["risk_category"].value_counts().reset_index()
                risk_counts.columns = ["Risk Tier", "Count"]
                color_map = {"High Risk": "#EF4444", "Medium Risk": "#F59E0B", "Low Risk": "#10B981"}
                fig_dist = px.bar(
                    risk_counts,
                    x="Risk Tier",
                    y="Count",
                    color="Risk Tier",
                    color_discrete_map=color_map,
                    text="Count",
                    title="Batch Risk Tier Distribution",
                )
                fig_dist.update_layout(
                    template="plotly_white",
                    showlegend=False,
                    height=300,
                    margin=dict(t=35, b=20),
                )
                st.plotly_chart(fig_dist, use_container_width=True)

                # Result table
                st.markdown("### 📋 Prediction Results & Recommended Actions")
                view_cols = [
                    c for c in [
                        "customerID", "risk_category", "churn_probability", "priority",
                        "primary_strategy", "primary_action", "expected_retention_value",
                        "estimated_cost", "revenue_protected", "Contract", "MonthlyCharges", "tenure",
                    ] if c in recs_df.columns
                ]
                if not view_cols:
                    view_cols = recs_df.columns.tolist()

                display_res = recs_df[view_cols].copy()
                if "churn_probability" in display_res:
                    display_res["churn_probability"] = display_res["churn_probability"].apply(
                        lambda p: f"{float(p) * 100:.1f}%"
                    )
                for money_col in ["expected_retention_value", "estimated_cost", "revenue_protected", "MonthlyCharges"]:
                    if money_col in display_res:
                        display_res[money_col] = display_res[money_col].apply(format_currency)

                st.dataframe(display_res, use_container_width=True, height=450, hide_index=True)

                # Risk filter for download
                risk_filter_opts = ["All", "High Risk", "Medium Risk", "Low Risk"]
                selected_risk_filter = st.selectbox(
                    "Filter export by risk tier:", risk_filter_opts, index=0
                )
                if selected_risk_filter == "All":
                    export_df = recs_df
                else:
                    export_df = recs_df[recs_df["risk_category"] == selected_risk_filter]

                st.markdown("### 📥 Download Options")
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.download_button(
                        label="⬇️ Export Complete Predictions (CSV)",
                        data=recs_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"batch_predictions_{file_source_name}",
                        mime="text/csv",
                        type="primary",
                        use_container_width=True,
                    )
                with ec2:
                    st.download_button(
                        label=f"⬇️ Export Filtered Roster ({len(export_df):,} accounts — {selected_risk_filter})",
                        data=export_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"filtered_{selected_risk_filter.lower().replace(' ', '_')}_{file_source_name}",
                        mime="text/csv",
                        use_container_width=True,
                    )

    except Exception as exc:
        st.error(f"Error processing file: {exc}")
        st.exception(exc)

else:
    st.info("⬆️ Upload a CSV file or use the **Load Demo Data** button above to test batch prediction.")

    # Show expected schema
    with st.expander("📐 Expected Raw Input CSV Schema", expanded=False):
        schema_df = pd.DataFrame([
            {"Column": "customerID", "Type": "string", "Required": "Yes", "Example": "7590-VHVEG"},
            {"Column": "gender", "Type": "string", "Required": "Yes", "Example": "Female"},
            {"Column": "SeniorCitizen", "Type": "int (0/1)", "Required": "Yes", "Example": "0"},
            {"Column": "Partner", "Type": "Yes/No", "Required": "Yes", "Example": "Yes"},
            {"Column": "Dependents", "Type": "Yes/No", "Required": "Yes", "Example": "No"},
            {"Column": "tenure", "Type": "int", "Required": "Yes", "Example": "1"},
            {"Column": "PhoneService", "Type": "Yes/No", "Required": "Yes", "Example": "No"},
            {"Column": "MultipleLines", "Type": "string", "Required": "Yes", "Example": "No phone service"},
            {"Column": "InternetService", "Type": "string", "Required": "Yes", "Example": "DSL"},
            {"Column": "OnlineSecurity", "Type": "string", "Required": "Yes", "Example": "No"},
            {"Column": "OnlineBackup", "Type": "string", "Required": "Yes", "Example": "Yes"},
            {"Column": "DeviceProtection", "Type": "string", "Required": "Yes", "Example": "No"},
            {"Column": "TechSupport", "Type": "string", "Required": "Yes", "Example": "No"},
            {"Column": "StreamingTV", "Type": "string", "Required": "Yes", "Example": "No"},
            {"Column": "StreamingMovies", "Type": "string", "Required": "Yes", "Example": "No"},
            {"Column": "Contract", "Type": "string", "Required": "Yes", "Example": "Month-to-month"},
            {"Column": "PaperlessBilling", "Type": "Yes/No", "Required": "Yes", "Example": "Yes"},
            {"Column": "PaymentMethod", "Type": "string", "Required": "Yes", "Example": "Electronic check"},
            {"Column": "MonthlyCharges", "Type": "float", "Required": "Yes", "Example": "29.85"},
            {"Column": "TotalCharges", "Type": "float", "Required": "Yes", "Example": "29.85"},
        ])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

render_footer()
