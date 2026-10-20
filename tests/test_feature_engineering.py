"""Tests for feature engineering transformations."""
import pytest
import pandas as pd
from src.feature_engineering import engineer_features, prepare_features_and_target


@pytest.fixture
def raw_customer_df():
    return pd.DataFrame([{
        "customerID": "TEST-100",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 6,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 570.0,
        "Churn": "Yes"
    }])


def test_feature_engineering_creates_expected_columns(raw_customer_df):
    eng = engineer_features(raw_customer_df)
    assert "tenure_group" in eng.columns
    assert eng.loc[0, "tenure_group"] == "0-12m"
    assert "num_services" in eng.columns
    # 1 phone + 1 multiple lines + 1 fiber optic + 1 streaming TV + 1 streaming movies = 5
    assert eng.loc[0, "num_services"] == 5
    assert eng.loc[0, "is_month_to_month"] == "Yes"
    assert eng.loc[0, "is_electronic_check"] == "Yes"
    assert eng.loc[0, "has_fiber_optic"] == "Yes"
    assert eng.loc[0, "is_senior_without_dependents"] == "Yes"


def test_zero_tenure_no_division_by_zero():
    df = pd.DataFrame([{
        "tenure": 0,
        "MonthlyCharges": 50.0,
        "TotalCharges": 0.0
    }])
    eng = engineer_features(df)
    assert not eng["charges_ratio"].isna().any()
    assert not eng["monthly_to_total_ratio"].isna().any()


def test_target_separation_prevents_leakage(raw_customer_df):
    X, y, ids = prepare_features_and_target(raw_customer_df)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert y is not None
    assert int(y.iloc[0]) == 1
    assert ids.iloc[0] == "TEST-100"
