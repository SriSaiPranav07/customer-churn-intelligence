"""Page 5: Model Performance & Comparative Evaluation.

Rigorous evaluation on held-out test data (ROC-AUC, PR-AUC, Calibration, Profit Curve, Fairness).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_loader import load_telco_data
from src.evaluation import (
    compute_comprehensive_metrics,
    plot_confusion_matrix_interactive,
    plot_pr_curve,
    plot_profit_curve,
    plot_reliability_curve,
    plot_roc_curve,
)
from src.explainability import get_shap_explainer
from src.ui_components import inject_custom_css, render_footer, render_sidebar_filters
from src.utils import MODELS_DIR, format_currency

st.set_page_config(page_title="Model Performance | Churn Intelligence", page_icon="📈", layout="wide")
inject_custom_css()
high_thresh, med_thresh = render_sidebar_filters()

st.title("📈 Model Performance & Evaluation")
st.markdown("Rigorous benchmarking across Logistic Regression, Random Forest, and XGBoost with 5-fold cross-validation, probability calibration, threshold optimization, and fairness audits.")

# Load saved metrics and cache (strictly without background auto-training)
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
CACHE_PATH = MODELS_DIR / "test_evaluation_cache.joblib"

if not METRICS_PATH.exists() or not CACHE_PATH.exists():
    st.error("Model artifacts or evaluation cache missing. Please execute `python src/train.py` from the project root to generate them.")
    st.stop()

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

# Champion Model Callout
champ_name = eval_data.get("metadata", {}).get("champion_model_name", eval_data.get("champion_model", "XGBoost"))
champ_details = eval_data.get("models", {}).get(champ_name, {})
champ_rationale = eval_data.get("metadata", {}).get("champion_rationale", eval_data.get("champion_model_rationale", ""))

st.markdown(f"""
    <div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-left: 5px solid #22C55E; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <h3 style="margin: 0; color: #166534;">🏆 Production Champion: {champ_name}</h3>
            <span style="background-color: #DCFCE7; color: #15803D; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem;">
                5-Fold CV Selected
            </span>
        </div>
        <p style="margin: 0; color: #14532D; font-size: 0.95rem; line-height: 1.5;">
            {champ_rationale}
        </p>
    </div>
""", unsafe_allow_html=True)

# Section 1: Comparative Metric Table
st.subheader("1. Cross-Validation & Held-Out Test Benchmarks")

comparison_rows = []
for name, details in eval_data["models"].items():
    dm = details["default_metrics"]
    opt = details.get("optimal_metrics", {})
    cv = details.get("cv_metrics", {})
    ci = details.get("bootstrap_95ci", {})
    ci_roc = [ci.get("roc_auc", {}).get("ci_lower", dm["roc_auc"]), ci.get("roc_auc", {}).get("ci_upper", dm["roc_auc"])]
    ci_pr = [ci.get("pr_auc", {}).get("ci_lower", dm["pr_auc"]), ci.get("pr_auc", {}).get("ci_upper", dm["pr_auc"])]
    opt_thresh = details.get("optimal_threshold", 0.5)

    comparison_rows.append({
        "Model": name,
        "CV ROC-AUC (Mean ± Std)": f"{cv.get('roc_auc_mean', 0):.4f} ± {cv.get('roc_auc_std', 0):.4f}",
        "Test ROC-AUC [95% CI]": f"{dm['roc_auc']:.4f} [{ci_roc[0]:.3f}, {ci_roc[1]:.3f}]",
        "Test PR-AUC [95% CI]": f"{dm['pr_auc']:.4f} [{ci_pr[0]:.3f}, {ci_pr[1]:.3f}]",
        "F1 (Cutoff=0.50)": f"{dm['f1']:.4f}",
        "F1 (Optimal Cutoff)": f"{opt.get('f1', dm['f1']):.4f} (@ {opt_thresh:.2f})",
        "Recall": f"{dm['recall']:.4f}",
        "Precision": f"{dm['precision']:.4f}",
        "Brier Score": f"{dm['brier_score']:.4f}"
    })

df_comp = pd.DataFrame(comparison_rows).set_index("Model")
st.dataframe(df_comp, use_container_width=True)

st.caption("Note: Preprocessing, scaling, and hyperparameter tuning were fit strictly on the 80% training split via unified Pipeline architectures to guarantee zero data leakage. Decision thresholds were selected on training cross-validation and evaluated once on the held-out test split.")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: ROC and Precision-Recall Curves
st.subheader("2. Discrimination Diagnostics: ROC & PR Curves")
curve_col1, curve_col2 = st.columns(2)

with curve_col1:
    fig_roc = plot_roc_curve(models_plot_dict)
    st.plotly_chart(fig_roc, use_container_width=True)

with curve_col2:
    fig_pr = plot_pr_curve(models_plot_dict)
    st.plotly_chart(fig_pr, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Probability Calibration & Reliability Curve
st.subheader("3. Probability Calibration & Reliability Assessment")
st.markdown("Raw tree model scores frequently exhibit probability distortion near 0 and 1. We calibrate predictions using Isotonic / Sigmoid regression to ensure that a predicted probability of 70% corresponds to an observed 70% churn rate.")

rel_curves = test_cache.get("reliability_curves", {})
if rel_curves:
    rel_col1, rel_col2 = st.columns([1.3, 1])
    with rel_col1:
        fig_cal = plot_reliability_curve(rel_curves)
        st.plotly_chart(fig_cal, use_container_width=True)
    with rel_col2:
        st.markdown("#### Calibration Performance (Brier Score)")
        st.markdown("The Brier score measures the mean squared difference between predicted probabilities and actual outcomes (lower is superior):")
        brier_data = []
        for m_name, d in eval_data["models"].items():
            brier_data.append({
                "Model": m_name,
                "Brier Score": d["default_metrics"]["brier_score"],
                "Calibration Method": d.get("calibration_method", "None")
            })
        st.dataframe(pd.DataFrame(brier_data), hide_index=True, use_container_width=True)
        st.info("Calibrated probabilities ensure reliable expected-value calculations for downstream commercial interventions.")

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Business Value & Decision Threshold Optimization
st.subheader("4. Business Value Optimization & Profit Curves")
st.markdown("In real retention workflows, false positives and false negatives incur asymmetric costs. Sweeping decision thresholds identifies the cutoff that maximizes expected commercial net value.")

profit_curves = test_cache.get("profit_curve", {})

p_col1, p_col2 = st.columns([1.3, 1])

selected_eval_model = p_col2.selectbox(
    "Select Model for Threshold Analysis",
    list(preds_dict.keys()),
    index=list(preds_dict.keys()).index(champ_name) if champ_name in preds_dict else 0
)

# Profit curve is for the champion only (computed once during training)
opt_profit_thresh = float(profit_curves.get("optimal_threshold", 0.50))
opt_profit_val = float(profit_curves.get("max_profit", 0.0))

with p_col1:
    profit_data = profit_curves.get("data", [])
    if profit_data:
        profit_df_plot = pd.DataFrame(profit_data)
        fig_profit = plot_profit_curve(profit_df_plot, opt_profit_thresh, title=f"Expected Net Retention Profit: Champion ({champ_name})")
        st.plotly_chart(fig_profit, use_container_width=True)

with p_col2:
    st.markdown(f"**Max Expected Net Value:** `{format_currency(opt_profit_val)}` at threshold **`{opt_profit_thresh:.2f}`**.")

    threshold_slider = st.slider(
        "Decision Threshold Cutoff",
        min_value=0.05, max_value=0.95, value=float(opt_profit_thresh), step=0.01,
        help="Lower thresholds capture more churners (higher recall) at the cost of outreach budget (lower precision)."
    )

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

# Confusion Matrix for selected threshold
cm_fig = plot_confusion_matrix_interactive(
    live_metrics["confusion_matrix"],
    title=f"Held-Out Confusion Matrix: {selected_eval_model} (Threshold = {threshold_slider:.2f})"
)
st.plotly_chart(cm_fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 5: Demographic Fairness & Bias Audit
st.subheader("5. Demographic Fairness & Bias Audit")
st.markdown("Machine learning models in customer operations must not exhibit disparate impact or unequal opportunity across protected demographic segments.")

fairness_audit = eval_data.get("fairness_audit", {})
if fairness_audit:
    f_rows = []
    for sensitive_col, groups in fairness_audit.items():
        for group_val, m in groups.items():
            f_rows.append({
                "Sensitive Attribute": sensitive_col,
                "Subgroup": group_val,
                "Sample Count": m.get("count", m.get("sample_count", 0)),
                "Base Churn Rate": f"{m.get('churn_rate', m.get('base_churn_rate', 0))*100:.1f}%",
                "Recall (TPR)": f"{m.get('recall', m.get('tpr', 0))*100:.1f}%",
                "Precision": f"{m.get('precision', 0)*100:.1f}%",
                "Selection Rate": f"{m.get('selection_rate', 0)*100:.1f}%",
                "ROC-AUC": f"{m.get('roc_auc', 0):.4f}"
            })
    st.dataframe(pd.DataFrame(f_rows), hide_index=True, use_container_width=True)

ablation_info = eval_data.get("gender_ablation", eval_data.get("gender_ablation_test", {}))
if ablation_info:
    baseline_roc = ablation_info.get("baseline_test_roc_auc", ablation_info.get("roc_auc_with_gender", 0))
    ablated_roc = ablation_info.get("without_gender_test_roc_auc", ablation_info.get("roc_auc_without_gender", 0))
    delta_roc = ablation_info.get("delta_roc_auc", ablation_info.get("roc_auc_difference", ablated_roc - baseline_roc))
    conclusion = ablation_info.get("conclusion", "")
    st.markdown(f"""
        <div style="background-color: #F8FAFC; border-left: 4px solid #6366F1; padding: 1rem; border-radius: 6px; margin-top: 0.75rem;">
            <b>Feature Ablation Analysis (Gender Parity):</b><br>
            ROC-AUC with gender: <b>{baseline_roc:.4f}</b> vs without gender: <b>{ablated_roc:.4f}</b> (Δ = {delta_roc:+.4f}).<br>
            <span style="color: #475569; font-size: 0.9rem;">{conclusion}</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Section 6: Global SHAP Feature Importance
st.subheader("6. Global Feature Importance (SHAP)")
st.markdown("Mean absolute SHAP value across customer accounts, revealing the primary structural drivers of churn risk.")

@st.cache_data(show_spinner="Calculating global SHAP importances...")
def get_global_shap():
    df, _ = load_telco_data()
    explainer = get_shap_explainer()
    return explainer.get_global_importance(df, n_samples=350)

df_global_shap = get_global_shap()
if not df_global_shap.empty:
    top20 = df_global_shap.head(15).iloc[::-1]
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

render_footer()
