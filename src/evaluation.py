"""Evaluation metrics calculation and interactive Plotly visualization charts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
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
            "threshold": th,
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

    return float(best_thresh), pd.DataFrame(records)


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
