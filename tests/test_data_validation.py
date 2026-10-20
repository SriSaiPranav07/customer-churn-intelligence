"""Tests for data validation and schema integrity."""
import pytest
import pandas as pd
from src.data_validation import validate_raw_data, clean_total_charges, EXPECTED_RAW_COLUMNS


@pytest.fixture
def sample_valid_row():
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.20,
        "TotalCharges": "662.40",
        "Churn": "No"
    }


def test_valid_data_passes(sample_valid_row):
    df = pd.DataFrame([sample_valid_row])
    is_valid, errors, warnings, cleaned_df = validate_raw_data(df, is_training=True)
    assert is_valid is True
    assert len(errors) == 0
    assert float(cleaned_df.loc[0, "TotalCharges"]) == pytest.approx(662.40)


def test_empty_dataframe_rejected():
    df = pd.DataFrame()
    is_valid, errors, _, _ = validate_raw_data(df, is_training=True)
    assert is_valid is False
    assert "empty" in errors[0].lower()


def test_missing_required_column_fails(sample_valid_row):
    del sample_valid_row["tenure"]
    df = pd.DataFrame([sample_valid_row])
    is_valid, errors, _, _ = validate_raw_data(df, is_training=True)
    assert is_valid is False
    assert any("tenure" in err for err in errors)


def test_negative_numeric_fails(sample_valid_row):
    sample_valid_row["MonthlyCharges"] = -10.5
    df = pd.DataFrame([sample_valid_row])
    is_valid, errors, _, _ = validate_raw_data(df, is_training=True)
    assert is_valid is False
    assert any("MonthlyCharges" in err for err in errors)


def test_whitespace_total_charges_cleaned():
    df = pd.DataFrame([{
        "tenure": 0,
        "MonthlyCharges": 25.0,
        "TotalCharges": "   "
    }])
    cleaned = clean_total_charges(df)
    assert cleaned.loc[0, "TotalCharges"] == 0.0
