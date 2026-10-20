"""Feature engineering pipeline for churn prediction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from src.utils import logger

NUMERICAL_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "charges_ratio",
    "monthly_to_total_ratio",
    "cost_per_service"
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
    "has_tech_support_or_security",
    "has_streaming",
    "is_month_to_month",
    "is_electronic_check",
    "has_fiber_optic",
    "is_senior_without_dependents"
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic, reproducible feature engineering.
    Can be called on both training and raw new inference customer data.
    """
    df = df.copy()

    # Ensure required numeric columns are present and clean
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # 1. Tenure grouping
    bins = [-1, 12, 24, 48, 60, 100]
    labels = ["0-12m", "13-24m", "25-48m", "49-60m", ">60m"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)

    # 2. Count active subscribed services
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    num_services = np.zeros(len(df), dtype=int)
    for col in service_cols:
        if col in df.columns:
            num_services += (df[col] == "Yes").astype(int)
    if "InternetService" in df.columns:
        num_services += (df["InternetService"].isin(["DSL", "Fiber optic"])).astype(int)

    df["num_services"] = num_services

    # 3. Financial and billing ratios
    # charges_ratio: checks if customer paid more/less than base rate * tenure (promo expiry indicator)
    expected_total = df["tenure"] * df["MonthlyCharges"]
    df["charges_ratio"] = np.where(
        expected_total > 0,
        np.round(df["TotalCharges"] / expected_total, 4),
        1.0
    )
    # Clip extreme outliers due to tenure=0
    df["charges_ratio"] = df["charges_ratio"].clip(0.1, 5.0)

    # Monthly-to-total ratio (higher in new customers)
    df["monthly_to_total_ratio"] = np.round(df["MonthlyCharges"] / (df["TotalCharges"] + 1.0), 4)

    # Cost per service
    df["cost_per_service"] = np.round(df["MonthlyCharges"] / (df["num_services"] + 1), 2)

    # 4. Behavioral & Risk Indicators
    n_rows = len(df)
    tech_col = df["TechSupport"] if "TechSupport" in df.columns else pd.Series(["No"] * n_rows, index=df.index)
    sec_col = df["OnlineSecurity"] if "OnlineSecurity" in df.columns else pd.Series(["No"] * n_rows, index=df.index)
    tech_sec = (tech_col == "Yes") | (sec_col == "Yes")
    df["has_tech_support_or_security"] = np.where(tech_sec, "Yes", "No")

    stv_col = df["StreamingTV"] if "StreamingTV" in df.columns else pd.Series(["No"] * n_rows, index=df.index)
    smov_col = df["StreamingMovies"] if "StreamingMovies" in df.columns else pd.Series(["No"] * n_rows, index=df.index)
    streaming = (stv_col == "Yes") | (smov_col == "Yes")
    df["has_streaming"] = np.where(streaming, "Yes", "No")

    contract_col = df["Contract"] if "Contract" in df.columns else pd.Series([""] * n_rows, index=df.index)
    df["is_month_to_month"] = np.where(contract_col == "Month-to-month", "Yes", "No")

    pay_col = df["PaymentMethod"] if "PaymentMethod" in df.columns else pd.Series([""] * n_rows, index=df.index)
    df["is_electronic_check"] = np.where(pay_col == "Electronic check", "Yes", "No")

    net_col = df["InternetService"] if "InternetService" in df.columns else pd.Series([""] * n_rows, index=df.index)
    df["has_fiber_optic"] = np.where(net_col == "Fiber optic", "Yes", "No")

    senior_col = df["SeniorCitizen"] if "SeniorCitizen" in df.columns else pd.Series([0] * n_rows, index=df.index)
    dep_col = df["Dependents"] if "Dependents" in df.columns else pd.Series(["No"] * n_rows, index=df.index)
    is_senior = pd.to_numeric(senior_col, errors="coerce").fillna(0).astype(int) == 1
    no_dep = dep_col == "No"
    df["is_senior_without_dependents"] = np.where(is_senior & no_dep, "Yes", "No")

    # Format SeniorCitizen to categorical string
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    return df


def prepare_features_and_target(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.Series]:
    """
    Separates customer ID, predictive features, and target.
    Returns:
        X: Feature DataFrame containing only engineered features
        y: Target series (0/1) if Churn exists, else None
        customer_ids: Customer ID series for business identification
    """
    df_engineered = engineer_features(df)
    
    # Extract IDs
    if "customerID" in df_engineered.columns:
        customer_ids = df_engineered["customerID"]
    else:
        customer_ids = pd.Series([f"CUST-{i:05d}" for i in range(len(df_engineered))], name="customerID")

    # Extract target if present
    y = None
    if "Churn" in df_engineered.columns:
        # Standardize target mapping
        y = df_engineered["Churn"].map({
            "Yes": 1, "No": 0,
            1: 1, 0: 0,
            "1": 1, "0": 0,
            True: 1, False: 0
        }).astype(int)

    # Select only valid features
    feature_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES if c in df_engineered.columns]
    X = df_engineered[feature_cols].copy()

    return X, y, customer_ids
