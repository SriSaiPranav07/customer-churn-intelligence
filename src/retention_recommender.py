"""Intelligent Retention Action Recommendation Engine (Mentor Challenge).

Maps customer churn risk, feature profiles, and SHAP drivers to targeted,
cost-effective business retention strategies with action tracking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, List, Optional
import pandas as pd

from src.utils import logger, ACTION_STATUS_OPTIONS

RETENTION_DISCLAIMER = (
    "Decision-Support Notice: Recommended retention actions are data-driven suggestions "
    "synthesized from customer risk profiles and feature sensitivities. They provide structured "
    "decision support for retention teams and do not constitute guaranteed churn prevention. "
    "Always apply human relationship management and business judgment before executing interventions."
)


def determine_recommendation_priority(prob: float) -> str:
    """Calculates operational priority based on churn risk probability."""
    if prob >= 0.70:
        return "Critical"
    elif prob >= 0.50:
        return "High"
    elif prob >= 0.35:
        return "Medium"
    return "Low"


def generate_customer_recommendation(
    customer_row: pd.Series,
    churn_probability: float,
    shap_risk_drivers: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Evaluates customer profile and SHAP risk drivers to generate tailored retention actions.
    Ensures discounts are only recommended when financially justified, avoiding blanket giveaways.
    """
    tenure = float(customer_row.get("tenure", 0))
    monthly_charges = float(customer_row.get("MonthlyCharges", 0.0))
    contract = str(customer_row.get("Contract", "Month-to-month"))
    payment_method = str(customer_row.get("PaymentMethod", ""))
    internet_service = str(customer_row.get("InternetService", ""))
    tech_support = str(customer_row.get("TechSupport", "No"))
    online_security = str(customer_row.get("OnlineSecurity", "No"))
    paperless = str(customer_row.get("PaperlessBilling", "No"))

    priority = determine_recommendation_priority(churn_probability)
    drivers_str = ""
    if shap_risk_drivers:
        drivers_str = " ".join([d.get("feature", "").lower() for d in shap_risk_drivers])

    # Rule Candidates (evaluated hierarchically)
    candidates: List[Dict[str, Any]] = []

    # Rule 1: High Monthly Cost & Price Sensitivity
    if (monthly_charges > 70.0) or ("monthly charges" in drivers_str) or ("cost per service" in drivers_str):
        if tenure > 12:
            candidates.append({
                "strategy": "Tier Optimization & Loyalty Plan",
                "action": "Offer a personalized plan review to right-size unused features, or enroll in a 10% loyalty rate on current tier.",
                "rationale": f"Customer pays above-average monthly charges (${monthly_charges:.2f}/mo) and has proven tenure ({tenure} mos). Right-sizing prevents price-shock churn.",
                "estimated_cost": 15.00,
                "revenue_protected": round(monthly_charges * 12 * 0.85, 2)
            })
        else:
            candidates.append({
                "strategy": "Value Realization & Feature Audit",
                "action": "Schedule a customer success onboarding session to demonstrate premium feature value and optimize active features.",
                "rationale": f"High charges (${monthly_charges:.2f}/mo) in early customer lifecycle ({tenure} mos) suggests risk of perceived low value-for-money without discounting.",
                "estimated_cost": 20.00,
                "revenue_protected": round(monthly_charges * 12 * 0.70, 2)
            })

    # Rule 2: Month-to-Month Contract Vulnerability
    if contract == "Month-to-month" or ("month-to-month" in drivers_str) or ("contract" in drivers_str):
        candidates.append({
            "strategy": "Contract Modernization Incentive",
            "action": "Invite customer to upgrade to a 1-year contract with a complimentary premium add-on (e.g. streaming perk or $5/month bill credit).",
            "rationale": "Month-to-month contracts lack switching barriers and exhibit 4x higher churn baseline. A 1-year commitment locks in annual recurring revenue.",
            "estimated_cost": 25.00,
            "revenue_protected": round(monthly_charges * 12 * 0.80, 2)
        })

    # Rule 3: Electronic Check & Billing Friction
    if ("electronic check" in payment_method.lower()) or ("electronic check" in drivers_str):
        candidates.append({
            "strategy": "AutoPay Transition & Billing Streamlining",
            "action": "Send guided invite to switch payment to Automated Bank Transfer or Credit Card with a one-time $10 statement credit.",
            "rationale": "Electronic check payments have the highest operational failure and manual friction rates in the telecom industry, directly triggering churn.",
            "estimated_cost": 10.00,
            "revenue_protected": round(monthly_charges * 6, 2)
        })

    # Rule 4: Fiber Optic Stability without Technical Protection
    if (internet_service == "Fiber optic") and (tech_support != "Yes" or online_security != "Yes"):
        candidates.append({
            "strategy": "VIP Technical Care Package",
            "action": "Provide 3 months of complimentary Tech Support and security diagnostic to resolve bandwidth or installation friction.",
            "rationale": "Fiber optic customers experience high churn when experiencing technical or speed inconsistencies without dedicated technical support.",
            "estimated_cost": 15.00,
            "revenue_protected": round(monthly_charges * 12 * 0.75, 2)
        })

    # Rule 5: New Customer Early Lifecycle Stabilization
    if tenure <= 6:
        candidates.append({
            "strategy": "Early Lifecycle Proactive Check-In",
            "action": "Trigger automated VIP welcome call from customer care specialist to resolve early friction or unanswered service questions.",
            "rationale": f"Customer is in the critical first {tenure} months where early cancellation probability peaks due to unaddressed onboarding questions.",
            "estimated_cost": 12.00,
            "revenue_protected": round(monthly_charges * 12 * 0.65, 2)
        })

    # Fallback Rule: General Customer Appreciation & Health Check
    if not candidates:
        candidates.append({
            "strategy": "Account Health Check & Appreciation",
            "action": "Send relationship check-in email thanking them for their loyalty with a self-service account optimization guide.",
            "rationale": "Maintains positive customer engagement and proactively surveys sentiment before latent dissatisfaction emerges.",
            "estimated_cost": 5.00,
            "revenue_protected": round(monthly_charges * 6 * 0.50, 2)
        })

    primary = candidates[0]
    secondary = candidates[1] if len(candidates) > 1 else None

    return {
        "customerID": str(customer_row.get("customerID", "UNKNOWN")),
        "churn_probability": round(churn_probability, 4),
        "priority": priority,
        "primary_strategy": primary["strategy"],
        "primary_action": primary["action"],
        "rationale": primary["rationale"],
        "estimated_cost": primary["estimated_cost"],
        "revenue_protected": primary["revenue_protected"],
        "secondary_strategy": secondary["strategy"] if secondary else "None",
        "secondary_action": secondary["action"] if secondary else "None",
        "action_status": "Planned",
        "disclaimer": RETENTION_DISCLAIMER
    }


def generate_batch_recommendations(
    predicted_df: pd.DataFrame,
    shap_explainer=None
) -> pd.DataFrame:
    """
    Generates intelligent retention recommendations for all rows in a predicted customer DataFrame.
    Returns a unified actionable retention table.
    """
    records = []
    for _, row in predicted_df.iterrows():
        prob = float(row.get("churn_probability", 0.0))
        # Top SHAP drivers if available
        shap_drivers = None
        if shap_explainer is not None:
            try:
                res = shap_explainer.explain_single_customer(row)
                shap_drivers = res.get("risk_drivers", [])
            except Exception:
                shap_drivers = None

        rec = generate_customer_recommendation(row, prob, shap_drivers)
        # Add basic customer profile context
        rec["tenure"] = int(row.get("tenure", 0))
        rec["MonthlyCharges"] = float(row.get("MonthlyCharges", 0.0))
        rec["Contract"] = str(row.get("Contract", "Month-to-month"))
        rec["PaymentMethod"] = str(row.get("PaymentMethod", ""))
        rec["InternetService"] = str(row.get("InternetService", ""))
        rec["risk_category"] = str(row.get("risk_category", "Low Risk"))
        records.append(rec)

    rec_df = pd.DataFrame(records)
    return rec_df
