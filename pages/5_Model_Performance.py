"""Page 5: Model Performance & Comparative Evaluation.

Rigorous evaluation on held-out test data (ROC-AUC, PR-AUC, Confusion Matrix, Threshold Optimization).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils import MODELS_DIR, COLOR_PRIMARY
from src.evaluation import (
    plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix_interactive, compute_comprehensive_metrics
)
from src.explainability import get_shap_explainer
from src.data_loader import load_telco_data
from src.ui_components import inject_custom_css, render_sidebar_filters, render_footer

st.set_page_config(page_title="Model Performance | Churn Intelligence", page_icon="📈", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("📈 Model Performance & Evaluation")
st.markdown("Rigorous benchmarking across Logistic Regression, Random Forest, and XGBoost on held-out test data (N = 1,409).")

# Load saved metrics and cache with auto-training fallback
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CACHE_PATH = MODELS_DIR / "test_evaluation_cache.joblib"

if not METRICS_PATH.exists() or not CACHE_PATH.exists():
    with st.spinner("Training models and generating held-out evaluation cache..."):
        from src.train import train_and_evaluate_all_models
        train_and_evaluate_all_models()

with open(METRICS_PATH, "r") as f:
    eval_data = json.load(f)

test_cache = joblib.load(CACHE_PATH)
y_test = test_cache["y_test"]
preds_dict = test_cache["predictions"]

# Construct comparative model dictionary for curves
models_plot_dict = {}
for m_name, p_arr in preds_dict.items():
    m_metrics = eval_data["models"][m_name]["default_metrics"]
    models_plot_dict[m_name] = {
        "y_test": y_test,
        "y_pred_proba": p_arr,
        "metrics": m_metrics
    }

# Section 1: Comparative Metric Table
st.subheader("1. Held-Out Test Evaluation Matrix (Threshold = 0.50)")

comparison_rows = []
for name, details in eval_data["models"].items():
    dm = details["default_metrics"]
    opt = details["optimal_metrics"]
    comparison_rows.append({
        "Model": name,
        "Description": details["description"],
        "ROC-AUC": dm["roc_auc"],
        "PR-AUC": dm["pr_auc"],
        "Recall (Churn)": dm["recall"],
        "Precision (Churn)": dm["precision"],
        "F1-Score": dm["f1"],
        "Accuracy": dm["accuracy"],
        "Brier Score": dm["brier_score"]
    })

df_comp = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(
    df_comp.style.highlight_max(axis=0, subset=["ROC-AUC", "PR-AUC", "Recall (Churn)", "Precision (Churn)", "F1-Score", "Accuracy"], color="#DCFCE7")
                 .highlight_min(axis=0, subset=["Brier Score"], color="#DCFCE7")
                 .format("{:.4f}", subset=["ROC-AUC", "PR-AUC", "Recall (Churn)", "Precision (Churn)", "F1-Score", "Accuracy", "Brier Score"]),
    use_container_width=True
)

st.caption("Note: Metrics are calculated strictly on the held-out 20% test dataset. Training metrics are omitted to guarantee zero overfitting bias.")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: ROC and Precision-Recall Curves
st.subheader("2. ROC & Precision-Recall Curves")
curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    fig_roc = plot_roc_curve(models_plot_dict)
    st.plotly_chart(fig_roc, use_container_width=True)

with curve_col2:
    fig_pr = plot_pr_curve(models_plot_dict)
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Interactive Confusion Matrix & Decision Threshold Simulator
st.subheader("3. Interactive Decision Threshold Simulator")
st.markdown("In retention operations, a default 0.50 threshold is rarely optimal. Adjusting the cutoff balances proactive churn prevention against unnecessary discount costs.")

sim_m1, sim_m2 = st.columns([1, 1.3])

with sim_m1:
    selected_eval_model = st.selectbox(
        "Select Model for Threshold Analysis",
        list(preds_dict.keys()),
        index=list(preds_dict.keys()).index("Random Forest") if "Random Forest" in preds_dict else 0
    )

    threshold_slider = st.slider(
        "Decision Threshold (Cutoff)",
        min_value=0.10, max_value=0.90, value=0.42, step=0.02,
        help="Lower thresholds maximize Churn Recall (capturing more leaving customers). Higher thresholds maximize Precision."
    )

    # Dynamic metrics computation
    current_proba = preds_dict[selected_eval_model]
    live_metrics = compute_comprehensive_metrics(y_test, current_proba, threshold=threshold_slider)

    st.markdown(f"""
        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 1rem; margin-top: 0.5rem;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; font-size: 0.9rem;">
                <div>Recall (Sensitivity): <b>{live_metrics['recall']*100:.1f}%</b></div>
                <div>Precision: <b>{live_metrics['precision']*100:.1f}%</b></div>
                <div>F1-Score: <b>{live_metrics['f1']:.4f}</b></div>
                <div>Specificity: <b>{live_metrics['specificity']*100:.1f}%</b></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with sim_m2:
    cm_fig = plot_confusion_matrix_interactive(
        live_metrics["confusion_matrix"],
        title=f"Held-Out Confusion Matrix: {selected_eval_model} (Threshold = {threshold_slider:.2f})"
    )
    st.plotly_chart(cm_fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Global SHAP Feature Importance
st.subheader("4. Global Feature Importance (SHAP)")
st.markdown("Mean absolute SHAP value across customer accounts, revealing the global structural drivers of churn risk.")

@st.cache_data(show_spinner="Calculating global SHAP importances...")
def get_global_shap():
    df, _ = load_telco_data()
    explainer = get_shap_explainer()
    return explainer.get_global_importance(df, n_samples=350)

df_global_shap = get_global_shap()
if not df_global_shap.empty:
    top20 = df_global_shap.head(15).iloc[::-1]  # Reverse for horizontal chart
    fig_glob = px.bar(
        top20,
        x="mean_abs_shap",
        y="feature",
        orientation="h",
        text=top20["mean_abs_shap"].apply(lambda v: f"{v:.4f}"),
        color="mean_abs_shap",
        color_continuous_scale="Blues"
    )
    fig_glob.update_layout(
        template="plotly_white",
        xaxis_title="Mean |SHAP Value| (Average Model Impact)",
        yaxis_title="",
        height=480,
        coloraxis_showscale=False,
        margin=dict(t=20, b=20, l=10, r=10)
    )
    st.plotly_chart(fig_glob, use_container_width=True)

# Methodology callout
st.markdown("""
    <div class="callout-box">
        <b>Class Imbalance & Rigor:</b><br>
        The dataset has a baseline churn rate of 26.5% (approx 3:1 retention-to-churn ratio). Tree models were trained using <code>class_weight='balanced_subsample'</code> (Random Forest) and <code>scale_pos_weight=2.77</code> (XGBoost). This prevents majority-class bias and ensures the models maintain high sensitivity to customer churn signals.
    </div>
""", unsafe_allow_html=True)

render_footer()
