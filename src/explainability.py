"""Explainable AI (XAI) using SHAP for global and local churn interpretation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap

from src.feature_engineering import prepare_features_and_target
from src.predict import get_predictor
from src.utils import logger, COLOR_HIGH_RISK, COLOR_LOW_RISK

CAUSATION_DISCLAIMER = (
    "Note on Interpretation: SHAP values quantify each feature's directional contribution "
    "to the model's mathematical output. High SHAP values indicate strong statistical association "
    "and model influence, NOT proven direct physical causation. Interventions should be validated "
    "with domain judgment and controlled business experiments."
)


def clean_feature_label(name: str) -> str:
    """Converts pipeline column transformer names to clean, human-readable labels."""
    # Strip prefix like num__ or cat__
    if "__" in name:
        name = name.split("__", 1)[1]
    
    # Clean standard categories
    name = name.replace("Contract_", "Contract: ")
    name = name.replace("PaymentMethod_", "Payment: ")
    name = name.replace("InternetService_", "Internet: ")
    name = name.replace("tenure_group_", "Tenure Group: ")
    name = name.replace("OnlineSecurity_", "Online Security: ")
    name = name.replace("TechSupport_", "Tech Support: ")
    name = name.replace("PaperlessBilling_", "Paperless Billing: ")
    name = name.replace("is_month_to_month_", "Month-to-Month: ")
    name = name.replace("is_electronic_check_", "Electronic Check: ")
    name = name.replace("has_fiber_optic_", "Fiber Optic: ")
    name = name.replace("num_services", "Number of Active Services")
    name = name.replace("MonthlyCharges", "Monthly Charges ($)")
    name = name.replace("TotalCharges", "Total Charges ($)")
    name = name.replace("tenure", "Tenure (Months)")
    name = name.replace("charges_ratio", "Charges-to-Tenure Ratio")
    name = name.replace("cost_per_service", "Avg Cost per Service ($)")
    return name


class ChurnSHAPExplainer:
    """Manages TreeExplainer and generates local and global explanations."""

    def __init__(self, predictor=None):
        self.predictor = predictor or get_predictor()
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes TreeExplainer for the champion model."""
        if not self.predictor.is_ready():
            logger.warning("Predictor not ready for SHAP initialization.")
            return
        try:
            self.explainer = shap.TreeExplainer(self.predictor.model)
            logger.info("SHAP TreeExplainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TreeExplainer: {e}")

    def explain_single_customer(self, customer_row: pd.Series) -> Dict[str, Any]:
        """
        Calculates local SHAP explanation for an individual customer.
        Returns top risk factors, top protective factors, and a plain-English explanation.
        """
        if self.explainer is None:
            self._init_explainer()
            if self.explainer is None:
                return {
                    "error": "SHAP explainer could not be initialized.",
                    "narrative": "Explanation unavailable.",
                    "disclaimer": CAUSATION_DISCLAIMER
                }

        # Convert row to DataFrame
        df_single = pd.DataFrame([customer_row])
        X, _, _ = prepare_features_and_target(df_single)
        X_proc = self.predictor.preprocessor.transform(X)

        raw_shap = self.explainer.shap_values(X_proc)

        # For binary classifiers in Random Forest, shap_values can be a list [class 0, class 1]
        # or a 3D numpy array (samples, features, classes)
        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            shap_vec = raw_shap[1][0]  # Positive class (Churn)
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            shap_vec = raw_shap[0, :, 1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 2:
            shap_vec = raw_shap[0]
        else:
            shap_vec = np.array(raw_shap).flatten()

        feature_names = self.predictor.feature_names
        cleaned_names = [clean_feature_label(f) for f in feature_names]

        # Pair features with values and shap contributions
        records = []
        for orig_name, clean_name, s_val in zip(feature_names, cleaned_names, shap_vec):
            records.append({
                "feature": clean_name,
                "raw_feature": orig_name,
                "shap_value": float(s_val),
                "abs_shap": abs(float(s_val))
            })

        df_shap = pd.DataFrame(records).sort_values("shap_value", ascending=False)

        # Drivers increasing churn risk (positive SHAP)
        risk_drivers = df_shap[df_shap["shap_value"] > 0.001].head(4).to_dict(orient="records")

        # Protective factors reducing churn risk (negative SHAP)
        protective_factors = df_shap[df_shap["shap_value"] < -0.001].tail(4).to_dict(orient="records")
        # Reverse to show most protective first
        protective_factors.reverse()

        # Build plain-English narrative
        narrative = self._generate_customer_narrative(customer_row, risk_drivers, protective_factors)

        return {
            "risk_drivers": risk_drivers,
            "protective_factors": protective_factors,
            "narrative": narrative,
            "all_shap": df_shap,
            "disclaimer": CAUSATION_DISCLAIMER
        }

    def _generate_customer_narrative(
        self,
        customer_row: pd.Series,
        risk_drivers: List[Dict[str, Any]],
        protective_factors: List[Dict[str, Any]]
    ) -> str:
        """Constructs an executive plain-English summary of the customer's risk profile."""
        tenure = customer_row.get("tenure", "N/A")
        contract = customer_row.get("Contract", "N/A")
        monthly = customer_row.get("MonthlyCharges", "N/A")
        internet = customer_row.get("InternetService", "N/A")

        parts = []
        parts.append(f"Customer has been active for **{tenure} months** on a **{contract}** plan with monthly charges of **${monthly}**.")

        if risk_drivers:
            top_risks = [f"**{d['feature']}** (+{d['shap_value']:.2f})" for d in risk_drivers[:3]]
            parts.append(f"The primary factors driving churn risk upward are: {', '.join(top_risks)}.")
        else:
            parts.append("No strong positive churn risk accelerators were identified.")

        if protective_factors:
            top_protect = [f"**{p['feature']}** ({p['shap_value']:.2f})" for p in protective_factors[:2]]
            parts.append(f"Conversely, churn risk is moderated by customer strengths including: {', '.join(top_protect)}.")

        return " ".join(parts)

    def get_global_importance(self, df_sample: pd.DataFrame, n_samples: int = 300) -> pd.DataFrame:
        """Computes global feature importance based on mean absolute SHAP values across a representative sample."""
        if self.explainer is None:
            self._init_explainer()
            if self.explainer is None:
                return pd.DataFrame()

        sample = df_sample.head(n_samples)
        X, _, _ = prepare_features_and_target(sample)
        X_proc = self.predictor.preprocessor.transform(X)

        raw_shap = self.explainer.shap_values(X_proc)

        if isinstance(raw_shap, list) and len(raw_shap) == 2:
            matrix = raw_shap[1]
        elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
            matrix = raw_shap[:, :, 1]
        else:
            matrix = raw_shap

        mean_abs = np.mean(np.abs(matrix), axis=0)
        feature_names = [clean_feature_label(f) for f in self.predictor.feature_names]

        df_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_importance


def plot_customer_shap_waterfall(explanation: Dict[str, Any], title: str = "Key Drivers Influencing Churn Probability") -> go.Figure:
    """Generates an interactive horizontal bar chart depicting local SHAP impact."""
    drivers = explanation.get("risk_drivers", [])[:5]
    protective = explanation.get("protective_factors", [])[:5]

    combined = protective + drivers
    if not combined:
        return go.Figure()

    df = pd.DataFrame(combined)
    df["color"] = df["shap_value"].apply(lambda v: COLOR_HIGH_RISK if v > 0 else COLOR_LOW_RISK)
    df["direction"] = df["shap_value"].apply(lambda v: "Increases Risk" if v > 0 else "Reduces Risk")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"],
        x=df["shap_value"],
        orientation="h",
        marker_color=df["color"],
        text=[f"{v:+.3f}" for v in df["shap_value"]],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:.4f}<extra></extra>"
    ))

    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (Impact on Model Churn Output)",
        yaxis_title="",
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=40, b=30)
    )
    return fig


_global_explainer: Optional[ChurnSHAPExplainer] = None


def get_shap_explainer() -> ChurnSHAPExplainer:
    """Singleton getter for SHAP explainer."""
    global _global_explainer
    if _global_explainer is None:
        _global_explainer = ChurnSHAPExplainer()
    return _global_explainer
