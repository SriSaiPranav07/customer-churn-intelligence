"""Tests for model prediction pipeline and risk classification."""
import pytest
import pandas as pd
from src.predict import get_predictor
from src.data_loader import load_telco_data


@pytest.fixture(scope="module")
def predictor():
    pred = get_predictor()
    assert pred.is_ready(), "Predictor model/preprocessor failed to initialize."
    return pred


def test_predictor_runs_on_sample(predictor):
    raw_df, _ = load_telco_data()
    sample = raw_df.head(10).copy()
    results = predictor.predict(sample)

    assert len(results) == 10
    assert "churn_probability" in results.columns
    assert "risk_category" in results.columns
    assert "prediction_timestamp" in results.columns

    # Check probability bounds
    probs = results["churn_probability"]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Check risk categories
    valid_categories = {"High Risk", "Medium Risk", "Low Risk"}
    assert set(results["risk_category"]).issubset(valid_categories)


def test_risk_category_assignment(predictor):
    raw_df, _ = load_telco_data()
    sample = raw_df.head(20).copy()
    results = predictor.predict(sample, high_threshold=0.60, med_threshold=0.35)

    for _, row in results.iterrows():
        p = row["churn_probability"]
        cat = row["risk_category"]
        if p >= 0.60:
            assert cat == "High Risk"
        elif p >= 0.35:
            assert cat == "Medium Risk"
        else:
            assert cat == "Low Risk"
