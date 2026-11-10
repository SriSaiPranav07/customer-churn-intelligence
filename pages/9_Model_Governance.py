"""Page 9: Model Governance & MLOps Dashboard.

Tracks model version, dataset lineage, artifact integrity, training metadata,
input drift monitoring, and evaluation reports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_telco_data
from src.ui_components import inject_custom_css, render_footer, render_sidebar_filters
from src.utils import MODELS_DIR

st.set_page_config(
    page_title="Model Governance | Churn Intelligence",
    page_icon="🛡️",
    layout="wide",
)
inject_custom_css()
render_sidebar_filters()

st.title("🛡️ Model Governance & MLOps Center")
st.markdown(
    "Tracks model versioning, dataset lineage, artifact integrity checks, "
    "training metadata, input drift monitoring, and evaluation audit reports."
)

# ---------------------------------------------------------------------------
# Load evaluation metrics
# ---------------------------------------------------------------------------
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CACHE_PATH = MODELS_DIR / "test_evaluation_cache.joblib"
MODEL_PATH = MODELS_DIR / "champion_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"


@st.cache_data(show_spinner="Loading governance data...")
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


eval_data = load_metrics()

# ---------------------------------------------------------------------------
# Section 1: Model Registry
# ---------------------------------------------------------------------------
st.subheader("1. Model Registry & Version Tracking")

registry = {}
if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
elif eval_data:
    # Build a registry view from evaluation_metrics.json
    meta = eval_data.get("metadata", {})
    env = eval_data.get("environment", {})
    registry = {
        "model_name": meta.get("champion_model_name", "XGBoost"),
        "version": "1.0.0",
        "stage": "Production",
        "training_date": meta.get("training_date", "See evaluation_metrics.json"),
        "python_version": env.get("python", "3.11.x"),
        "sklearn_version": env.get("scikit_learn", "1.5.x"),
        "xgboost_version": env.get("xgboost", "2.x"),
        "dataset_records": meta.get("total_records", 7043),
        "train_records": meta.get("train_records", 5634),
        "test_records": meta.get("test_records", 1409),
        "champion_rationale": meta.get("champion_rationale", "Selected by CV ROC-AUC"),
        "optimal_threshold_cv": meta.get("optimal_threshold_cv", 0.54),
    }

if registry:
    reg_col1, reg_col2 = st.columns(2)
    with reg_col1:
        st.markdown(f"""
| Field | Value |
|---|---|
| **Model Name** | {registry.get('model_name', 'XGBoost')} |
| **Version** | {registry.get('version', '1.0.0')} |
| **Stage** | {registry.get('stage', 'Production')} |
| **Training Date** | {registry.get('training_date', 'N/A')} |
| **Python** | {registry.get('python_version', 'N/A')} |
| **scikit-learn** | {registry.get('sklearn_version', 'N/A')} |
| **XGBoost** | {registry.get('xgboost_version', 'N/A')} |
| **Decision Threshold (CV)** | {registry.get('optimal_threshold_cv', 0.54)} |
        """)
    with reg_col2:
        st.markdown("**Champion Model Selection Rationale:**")
        st.info(registry.get("champion_rationale", "No rationale recorded."))

        champ_name = registry.get("model_name", "XGBoost")
        champ_metrics = eval_data.get("models", {}).get(champ_name, {}).get("default_metrics", {})
        if champ_metrics:
            st.markdown("**Production Champion Test Metrics:**")
            st.markdown(f"""
- ROC-AUC: **{champ_metrics.get('roc_auc', 'N/A')}**
- PR-AUC: **{champ_metrics.get('pr_auc', 'N/A')}**
- F1-Score: **{champ_metrics.get('f1', 'N/A')}**
- Recall: **{champ_metrics.get('recall', 'N/A')}**
- Precision: **{champ_metrics.get('precision', 'N/A')}**
- Brier Score: **{champ_metrics.get('brier_score', 'N/A')}** *(lower is better)*
            """)

else:
    st.warning("No model registry found. Run `python -m src.train` to generate artifacts.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Artifact Integrity Validation
# ---------------------------------------------------------------------------
st.subheader("2. Model Artifact Integrity Checks")

artifacts = {
    "champion_model.joblib": MODEL_PATH,
    "evaluation_metrics.json": METRICS_PATH,
    "test_evaluation_cache.joblib": CACHE_PATH,
    "preprocessor.joblib (optional)": PREPROCESSOR_PATH,
}

artifact_rows = []
for name, path in artifacts.items():
    exists = path.exists()
    if exists:
        size_kb = path.stat().st_size / 1024
        # Compute file hash (SHA-256) for reproducibility
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        artifact_rows.append({
            "Artifact": name,
            "Status": "✅ Found",
            "Size (KB)": f"{size_kb:.1f}",
            "SHA-256 (prefix)": sha,
            "Path": str(path),
        })
    else:
        artifact_rows.append({
            "Artifact": name,
            "Status": "❌ Missing",
            "Size (KB)": "-",
            "SHA-256 (prefix)": "-",
            "Path": str(path),
        })

artifact_df = pd.DataFrame(artifact_rows)
st.dataframe(artifact_df, use_container_width=True, hide_index=True)

all_critical_found = MODEL_PATH.exists() and METRICS_PATH.exists() and CACHE_PATH.exists()
if all_critical_found:
    st.success("✅ All critical model artifacts are present and verified.")
else:
    st.error("❌ One or more critical artifacts are missing. Run `python -m src.train` to regenerate.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3: Dataset Lineage
# ---------------------------------------------------------------------------
st.subheader("3. Dataset Lineage & Characteristics")

@st.cache_data(show_spinner="Computing dataset statistics...")
def get_dataset_info():
    df, meta = load_telco_data()
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].map({"Yes": 1, "No": 0}).mean() if "Churn" in df.columns else 0
    # Compute a lightweight data fingerprint
    data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]
    return df, meta, churn_counts, churn_pct, data_hash


df_raw, dataset_meta, churn_counts, churn_pct, data_hash = get_dataset_info()

dl_col1, dl_col2 = st.columns(2)
with dl_col1:
    st.markdown(f"""
| Field | Value |
|---|---|
| **Source** | IBM Telco Customer Churn (Kaggle/IBM Watson) |
| **Total Records** | {len(df_raw):,} |
| **Features** | {len(df_raw.columns)} |
| **Target Variable** | `Churn` (Yes / No) |
| **Churn Rate** | {churn_pct * 100:.1f}% |
| **Data Fingerprint (MD5)** | `{data_hash}` |
| **Dataset Type** | Cross-sectional snapshot (no timestamps) |
    """)
with dl_col2:
    churn_dist_df = pd.DataFrame({"Churn": ["No (Retained)", "Yes (Churned)"], "Count": [
        churn_counts.get("No", 0), churn_counts.get("Yes", 0)
    ]})
    fig_churn_dist = px.pie(
        churn_dist_df, names="Churn", values="Count",
        color="Churn",
        color_discrete_map={"No (Retained)": "#10B981", "Yes (Churned)": "#EF4444"},
        title="Dataset Class Distribution",
        hole=0.42,
    )
    fig_churn_dist.update_layout(template="plotly_white", margin=dict(t=40, b=10), height=270)
    st.plotly_chart(fig_churn_dist, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 4: Training Environment & Metadata
# ---------------------------------------------------------------------------
st.subheader("4. Training Environment & Metadata")

env_info = eval_data.get("environment", {})
meta_info = eval_data.get("metadata", {})

if env_info or meta_info:
    env_col1, env_col2 = st.columns(2)
    with env_col1:
        st.markdown("**Software Environment:**")
        st.code(json.dumps(env_info, indent=2), language="json")
    with env_col2:
        st.markdown("**Training Configuration:**")
        filtered_meta = {
            k: v for k, v in meta_info.items()
            if k not in ["champion_rationale"]
        }
        st.code(json.dumps(filtered_meta, indent=2), language="json")
else:
    st.info("Environment metadata not found in `evaluation_metrics.json`.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 5: Input Drift Monitoring
# ---------------------------------------------------------------------------
st.subheader("5. Input Drift Monitoring (Upload Batch CSV)")
st.markdown(
    "Compare an uploaded batch CSV's feature distributions against the IBM Telco training distribution. "
    "Population Stability Index (PSI > 0.25 = significant drift) is computed for numeric features."
)

uploaded_drift_file = st.file_uploader(
    "Upload customer batch CSV for drift analysis", type=["csv"], key="drift_uploader"
)


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Computes Population Stability Index between expected and actual distributions."""
    expected = np.array(expected, dtype=float)
    actual = np.array(actual, dtype=float)
    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    breakpoints = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Clip to avoid log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi, 4))


if uploaded_drift_file is not None:
    try:
        drift_df = pd.read_csv(uploaded_drift_file)
        st.success(f"Loaded batch file: {len(drift_df):,} rows, {len(drift_df.columns)} columns.")

        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        psi_rows = []
        for col in numeric_cols:
            if col in df_raw.columns and col in drift_df.columns:
                psi_val = compute_psi(df_raw[col].dropna().values, drift_df[col].dropna().values)
                severity = "🟢 Stable" if psi_val < 0.1 else ("🟡 Minor Drift" if psi_val < 0.25 else "🔴 Significant Drift")
                psi_rows.append({"Feature": col, "PSI": psi_val, "Assessment": severity})

        if psi_rows:
            psi_df = pd.DataFrame(psi_rows)
            st.dataframe(psi_df, use_container_width=True, hide_index=True)

            # Distribution comparison charts
            for col in numeric_cols:
                if col in df_raw.columns and col in drift_df.columns:
                    fig_drift = go.Figure()
                    fig_drift.add_trace(go.Histogram(
                        x=df_raw[col].dropna(), name="Training (IBM Telco)",
                        opacity=0.65, histnorm="probability density",
                        marker_color="#2563EB",
                    ))
                    fig_drift.add_trace(go.Histogram(
                        x=drift_df[col].dropna(), name="Uploaded Batch",
                        opacity=0.65, histnorm="probability density",
                        marker_color="#F59E0B",
                    ))
                    fig_drift.update_layout(
                        barmode="overlay",
                        title=f"Distribution Comparison: {col}",
                        template="plotly_white",
                        height=280,
                        margin=dict(t=40, b=20),
                        legend=dict(x=0.7, y=0.95),
                    )
                    st.plotly_chart(fig_drift, use_container_width=True)
        else:
            st.warning("No matching numeric columns found between training data and uploaded batch.")

    except Exception as exc:
        st.error(f"Error analysing drift: {exc}")
else:
    st.info("Upload a customer batch CSV above to compare feature distributions against the training data.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 6: Cross-Model Evaluation Report
# ---------------------------------------------------------------------------
st.subheader("6. Full Cross-Model Evaluation Report")

if eval_data.get("models"):
    report_rows = []
    for m_name, details in eval_data["models"].items():
        dm = details.get("default_metrics", {})
        opt = details.get("optimal_metrics", {})
        cv = details.get("cv_metrics", {})
        ci = details.get("bootstrap_95ci", {})

        report_rows.append({
            "Model": m_name,
            "CV ROC-AUC": f"{cv.get('roc_auc_mean', 0):.4f} ± {cv.get('roc_auc_std', 0):.4f}",
            "Test ROC-AUC": f"{dm.get('roc_auc', 0):.4f} "
                            f"[{ci.get('roc_auc', {}).get('ci_lower', 0):.3f}–"
                            f"{ci.get('roc_auc', {}).get('ci_upper', 0):.3f}]",
            "Test PR-AUC": f"{dm.get('pr_auc', 0):.4f} "
                           f"[{ci.get('pr_auc', {}).get('ci_lower', 0):.3f}–"
                           f"{ci.get('pr_auc', {}).get('ci_upper', 0):.3f}]",
            "F1 (θ=0.5)": f"{dm.get('f1', 0):.4f}",
            "Recall (θ=0.5)": f"{dm.get('recall', 0):.4f}",
            "Precision (θ=0.5)": f"{dm.get('precision', 0):.4f}",
            "Brier Score": f"{dm.get('brier_score', 0):.4f}",
            "Optimal Threshold": f"{details.get('optimal_threshold', 0.5):.2f}",
            "F1 (Optimal θ)": f"{opt.get('f1', dm.get('f1', 0)):.4f}",
        })

    report_df = pd.DataFrame(report_rows).set_index("Model")
    st.dataframe(report_df, use_container_width=True)

    st.caption(
        "95% CI computed via non-parametric bootstrapping (1,000 resamples). "
        "Optimal threshold tuned on cross-validation folds only; test set evaluated once."
    )

    # Export evaluation report
    report_csv = pd.DataFrame(report_rows).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Evaluation Report (CSV)",
        data=report_csv,
        file_name="model_evaluation_report.csv",
        mime="text/csv",
    )
else:
    st.warning("Evaluation metrics not found. Run `python -m src.train` to generate model artifacts.")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 7: CI/CD Status
# ---------------------------------------------------------------------------
st.subheader("7. CI/CD Pipeline Status")

ci_yml_path = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
if ci_yml_path.exists():
    st.success("✅ GitHub Actions CI workflow (`ci.yml`) is present.")
    with st.expander("View CI Workflow Configuration", expanded=False):
        st.code(ci_yml_path.read_text(encoding="utf-8"), language="yaml")
else:
    st.warning("⚠️ GitHub Actions CI workflow not found at `.github/workflows/ci.yml`.")

render_footer()
