"""Evaluation metrics calculation, statistical uncertainty, fairness, and Plotly visualization charts."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes full set of binary classification metrics.
    Never fabricates metrics; all values are computed directly from actual held-out predictions.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    brier = float(brier_score_loss(y_true, y_pred_proba))

    # Churn class specificity (True Negative Rate)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "specificity": round(spec, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    min_recall: float = 0.70
) -> Tuple[float, pd.DataFrame]:
    """
    Finds the optimal threshold that balances precision and recall for customer retention,
    ensuring at least min_recall of churners are captured.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh = 0.5
    best_f1 = -1.0
    records = []

    for th in thresholds:
        metrics = compute_comprehensive_metrics(y_true, y_pred_proba, threshold=th)
        records.append({
            "threshold": round(float(th), 3),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"]
        })
        # Optimize for F1 given minimum recall threshold
        if metrics["recall"] >= min_recall and metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thresh = th

    # Fallback to pure F1 argmax if condition not met
    if best_f1 < 0:
        best_idx = np.argmax([r["f1"] for r in records])
        best_thresh = records[best_idx]["threshold"]
        best_f1 = records[best_idx]["f1"]

    return float(round(best_thresh, 3)), pd.DataFrame(records)


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Computes non-parametric bootstrap confidence intervals for ROC-AUC and PR-AUC.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    roc_scores = []
    pr_scores = []

    alpha_low = ((1.0 - ci) / 2.0) * 100
    alpha_high = (ci + (1.0 - ci) / 2.0) * 100

    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, n_samples)
        sample_true = y_true[idx]
        sample_proba = y_pred_proba[idx]
        if len(np.unique(sample_true)) < 2:
            continue
        roc_scores.append(roc_auc_score(sample_true, sample_proba))
        pr_scores.append(average_precision_score(sample_true, sample_proba))

    return {
        "roc_auc": {
            "mean": round(float(np.mean(roc_scores)), 4),
            "std": round(float(np.std(roc_scores)), 4),
            "ci_lower": round(float(np.percentile(roc_scores, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(roc_scores, alpha_high)), 4),
        },
        "pr_auc": {
            "mean": round(float(np.mean(pr_scores)), 4),
            "std": round(float(np.std(pr_scores)), 4),
            "ci_lower": round(float(np.percentile(pr_scores, alpha_low)), 4),
            "ci_upper": round(float(np.percentile(pr_scores, alpha_high)), 4),
        }
    }


def compute_reliability_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Calculates calibration reliability curve points and Brier score."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, y_pred_proba))
    return {
        "prob_pred": [round(float(p), 4) for p in prob_pred],
        "prob_true": [round(float(p), 4) for p in prob_true],
        "brier_score": round(brier, 4)
    }


def compute_fairness_analysis(
    df_eval: pd.DataFrame,
    y_true_col: str = "Churn_num",
    proba_col: str = "churn_probability",
    sensitive_cols: List[str] = ("gender", "SeniorCitizen"),
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Computes subgroup classification metrics across demographic attributes."""
    fairness_results = {}
    y_pred = (df_eval[proba_col] >= threshold).astype(int)
    y_true = df_eval[y_true_col].astype(int)

    for col in sensitive_cols:
        if col not in df_eval.columns:
            continue
        group_metrics = {}
        for group_val, grp_df in df_eval.groupby(col):
            grp_true = grp_df[y_true_col].astype(int).values
            grp_proba = grp_df[proba_col].values
            grp_pred = (grp_proba >= threshold).astype(int)

            if len(np.unique(grp_true)) > 1:
                auc = float(roc_auc_score(grp_true, grp_proba))
            else:
                auc = 0.5

            group_metrics[str(group_val)] = {
                "count": int(len(grp_df)),
                "churn_rate": round(float(np.mean(grp_true)), 4),
                "selection_rate": round(float(np.mean(grp_pred)), 4),
                "precision": round(float(precision_score(grp_true, grp_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(grp_true, grp_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(grp_true, grp_pred, zero_division=0)), 4),
                "roc_auc": round(auc, 4)
            }
        fairness_results[col] = group_metrics

    return fairness_results


def compute_profit_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    annual_revenue: float = 600.0,
    intervention_cost: float = 25.0,
    acceptance_rate: float = 0.40
) -> Tuple[float, float, pd.DataFrame]:
    """
    Computes net retention profit across decision thresholds:
    Expected Profit = TP * acceptance_rate * annual_revenue - (TP + FP) * intervention_cost.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    records = []
    best_profit = -float("inf")
    best_thresh = 0.5

    for th in thresholds:
        y_pred = (y_pred_proba >= th).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        flagged = tp + fp

        gain = tp * acceptance_rate * annual_revenue
        cost = flagged * intervention_cost
        net_profit = gain - cost

        records.append({
            "threshold": round(float(th), 3),
            "flagged": int(flagged),
            "tp": int(tp),
            "fp": int(fp),
            "cost": round(float(cost), 2),
            "gain": round(float(gain), 2),
            "net_profit": round(float(net_profit), 2)
        })

        if net_profit > best_profit:
            best_profit = net_profit
            best_thresh = th

    df_profit = pd.DataFrame(records)
    return float(round(best_thresh, 3)), float(round(best_profit, 2)), df_profit


def plot_profit_curve(
    profit_df: pd.DataFrame,
    optimal_thresh: float,
    title: str = "Threshold Profit Optimization Curve"
) -> go.Figure:
    """Generates an interactive Plotly profit curve showing optimal threshold."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profit_df["threshold"],
        y=profit_df["net_profit"],
        mode="lines",
        line=dict(color="#2563EB", width=3),
        name="Net Retention Profit ($)"
    ))

    # Mark optimal threshold
    best_row = profit_df[profit_df["threshold"] == optimal_thresh]
    if not best_row.empty:
        max_profit = best_row["net_profit"].values[0]
        fig.add_trace(go.Scatter(
            x=[optimal_thresh],
            y=[max_profit],
            mode="markers+text",
            marker=dict(color="#EF4444", size=11),
            text=[f"Optimal: {optimal_thresh:.2f} (${max_profit:,.0f})"],
            textposition="top center",
            name="Max Profit Cutoff"
        ))

    # Zero profit benchmark line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Break-even ($0)")

    fig.update_layout(
        title=title,
        xaxis_title="Classification Threshold",
        yaxis_title="Estimated Net Profit ($)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=35, b=25, l=10, r=10)
    )
    return fig


def plot_reliability_curve(
    reliability_dict: Dict[str, Dict[str, Any]],
    title: str = "Probability Calibration Reliability Curves"
) -> go.Figure:
    """Renders Plotly calibration curve comparing uncalibrated vs calibrated probabilities."""
    fig = go.Figure()

    for name, data in reliability_dict.items():
        brier = data.get("brier_score", 0.0)
        fig.add_trace(go.Scatter(
            x=data["prob_pred"],
            y=data["prob_true"],
            mode="lines+markers",
            name=f"{name} (Brier: {brier:.4f})",
            line=dict(width=2.5)
        ))

    # Perfectly calibrated 45-degree diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(dash="dash", color="gray", width=1.5)
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives (Empirical Churn)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(t=35, b=25, l=10, r=10)
    )
    return fig


def plot_roc_curve(models_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Generates comparative ROC Curves for all trained models."""
    fig = go.Figure()

    for name, res in models_dict.items():
        if "y_test" in res and "y_pred_proba" in res:
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_pred_proba"])
            auc_val = res["metrics"]["roc_auc"]
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC = {auc_val:.3f})",
                line=dict(width=2.5)
            ))

    # Diagonal benchmark line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Chance (AUC = 0.500)",
        line=dict(dash="dash", color="gray", width=1.5)
    ))

    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve Comparison",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Recall / Sensitivity)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.55, y=0.1, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig


def plot_pr_curve(models_dict: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Generates comparative Precision-Recall Curves for all trained models."""
    fig = go.Figure()

    for name, res in models_dict.items():
        if "y_test" in res and "y_pred_proba" in res:
            precision, recall, _ = precision_recall_curve(res["y_test"], res["y_pred_proba"])
            pr_auc = res["metrics"]["pr_auc"]
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"{name} (PR-AUC = {pr_auc:.3f})",
                line=dict(width=2.5)
            ))

    fig.update_layout(
        title="Precision-Recall Curve (Critical for Class Imbalance)",
        xaxis_title="Recall (Proportion of Churners Identified)",
        yaxis_title="Precision (Accuracy of Churn Alerts)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(x=0.55, y=0.85, bgcolor="rgba(255,255,255,0.8)")
    )
    return fig


def plot_confusion_matrix_interactive(cm_dict: Dict[str, int], title: str = "Confusion Matrix") -> go.Figure:
    """Creates a styled interactive Plotly heatmap for confusion matrix."""
    tn = cm_dict["tn"]
    fp = cm_dict["fp"]
    fn = cm_dict["fn"]
    tp = cm_dict["tp"]

    z = [[tn, fp], [fn, tp]]
    x = ["Retained (Predicted)", "Churn (Predicted)"]
    y = ["Retained (Actual)", "Churn (Actual)"]

    annotations = []
    labels = [
        [f"True Negative (TN)<br><b>{tn:,}</b><br>Retained correctly", f"False Positive (FP)<br><b>{fp:,}</b><br>False alarm"],
        [f"False Negative (FN)<br><b>{fn:,}</b><br>Missed churner", f"True Positive (TP)<br><b>{tp:,}</b><br>Detected churner"]
    ]

    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=labels[i][j],
                showarrow=False,
                font=dict(color="white" if (i == 0 and j == 0) or (i == 1 and j == 1) else "#1E293B", size=13)
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale="Blues",
        showscale=False
    ))
    fig.update_layout(
        title=title,
        annotations=annotations,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        width=500, height=400
    )
    return fig
