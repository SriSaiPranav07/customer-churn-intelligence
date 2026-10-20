"""Data validation schemas, range checks, and integrity enforcement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from src.utils import logger

# Expected columns for full training dataset
EXPECTED_RAW_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges"
]

# Categorical allowed values
VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    }
}


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the TotalCharges column:
    Converts whitespace strings to NaN and imputes based on tenure * MonthlyCharges.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        # Convert string to numeric
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # For new customers with tenure == 0 or missing TotalCharges, impute with 0 or MonthlyCharges
        mask_missing = df["TotalCharges"].isna()
        if mask_missing.any():
            imputed_vals = np.where(df.loc[mask_missing, "tenure"] == 0, 0.0, df.loc[mask_missing, "MonthlyCharges"])
            df.loc[mask_missing, "TotalCharges"] = imputed_vals
    return df


def validate_raw_data(df: pd.DataFrame, is_training: bool = True) -> Tuple[bool, List[str], List[str], pd.DataFrame]:
    """
    Validates input customer DataFrame.
    Returns:
        is_valid: bool
        errors: List of fatal validation errors
        warnings: List of non-fatal warnings
        cleaned_df: Validated and sanitized DataFrame
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or len(df) == 0:
        return False, ["Input dataset is empty."], [], pd.DataFrame()

    cleaned_df = df.copy()

    # 1. Check required columns
    required_cols = list(EXPECTED_RAW_COLUMNS)
    if is_training:
        required_cols.append("Churn")

    missing_cols = [col for col in required_cols if col not in cleaned_df.columns]
    if missing_cols:
        # Check if customerID is missing; if so, create a dummy identifier
        if "customerID" in missing_cols:
            warnings.append("Missing 'customerID' column. Generated synthetic identifiers.")
            cleaned_df["customerID"] = [f"CUST-{i:05d}" for i in range(len(cleaned_df))]
            missing_cols.remove("customerID")

        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")
            return False, errors, warnings, cleaned_df

    # 2. Check and clean TotalCharges
    cleaned_df = clean_total_charges(cleaned_df)

    # 3. Numeric range checks
    if (cleaned_df["tenure"] < 0).any():
        errors.append("Invalid values: 'tenure' cannot be negative.")
    if (cleaned_df["MonthlyCharges"] < 0).any():
        errors.append("Invalid values: 'MonthlyCharges' cannot be negative.")
    if (cleaned_df["TotalCharges"] < 0).any():
        errors.append("Invalid values: 'TotalCharges' cannot be negative.")

    # 4. Check for duplicate customer IDs
    dup_count = cleaned_df["customerID"].duplicated().sum()
    if dup_count > 0:
        warnings.append(f"Found {dup_count} duplicate customerID entries. Keeping the first occurrence.")
        cleaned_df = cleaned_df.drop_duplicates(subset=["customerID"], keep="first")

    # 5. Check categorical validity
    for col, allowed_vals in VALID_CATEGORIES.items():
        if col in cleaned_df.columns:
            unseen = set(cleaned_df[col].dropna().unique()) - allowed_vals
            if unseen:
                warnings.append(f"Column '{col}' contains unrecognized categories: {unseen}. Handled gracefully during encoding.")

    # 6. Check target column if training
    if is_training and "Churn" in cleaned_df.columns:
        valid_targets = {"Yes", "No", 1, 0, "1", "0", True, False}
        unseen_target = set(cleaned_df["Churn"].dropna().unique()) - valid_targets
        if unseen_target:
            errors.append(f"Column 'Churn' has invalid target values: {unseen_target}")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings, cleaned_df


def validate_uploaded_csv(df: pd.DataFrame) -> Tuple[bool, List[str], List[str], pd.DataFrame]:
    """Validates user-uploaded batch CSV file for inference."""
    return validate_raw_data(df, is_training=False)
