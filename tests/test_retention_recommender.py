"""Tests for retention action recommendation engine."""
import pandas as pd

from src.retention_recommender import (
    determine_recommendation_priority,
    generate_batch_recommendations,
    generate_customer_recommendation,
)


def test_priority_thresholds():
    assert determine_recommendation_priority(0.85) == "Critical"
    assert determine_recommendation_priority(0.60) == "High"
    assert determine_recommendation_priority(0.40) == "Medium"
    assert determine_recommendation_priority(0.20) == "Low"


def test_high_monthly_charges_generates_pricing_action():
    customer = pd.Series({
        "customerID": "TEST-PRICE",
        "tenure": 24,
        "MonthlyCharges": 99.50,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No"
    })
    rec = generate_customer_recommendation(customer, churn_probability=0.75)

    assert rec["priority"] == "Critical"
    assert "Tier Optimization" in rec["primary_strategy"] or "Contract Modernization" in rec["primary_strategy"]
    assert rec["estimated_cost"] > 0
    assert rec["revenue_protected"] > 0
    assert rec["action_status"] == "Planned"


def test_batch_recommendations():
    df = pd.DataFrame([
        {
            "customerID": "C-1",
            "tenure": 3,
            "MonthlyCharges": 85.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "churn_probability": 0.82,
            "risk_category": "High Risk"
        },
        {
            "customerID": "C-2",
            "tenure": 60,
            "MonthlyCharges": 20.0,
            "Contract": "Two year",
            "PaymentMethod": "Credit card (automatic)",
            "InternetService": "No",
            "TechSupport": "No internet service",
            "churn_probability": 0.05,
            "risk_category": "Low Risk"
        }
    ])
    recs_df = generate_batch_recommendations(df)
    assert len(recs_df) == 2
    assert "primary_strategy" in recs_df.columns
    assert "primary_action" in recs_df.columns
    assert recs_df.loc[0, "priority"] == "Critical"
    assert recs_df.loc[1, "priority"] == "Low"
    assert "expected_retention_value" in recs_df.columns


def test_expected_value_calculation_and_overrides():
    customer = pd.Series({
        "customerID": "EV-TEST",
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "InternetService": "Fiber optic",
        "TechSupport": "No"
    })
    # EV = P(churn) * acceptance_rate * annual_revenue - cost
    # 0.50 * 0.50 * 1000.0 - 50.0 = 250.0 - 50.0 = 200.0
    rec = generate_customer_recommendation(
        customer,
        churn_probability=0.50,
        cost_override=50.0,
        acceptance_rate=0.50,
        annual_revenue_basis=1000.0
    )
    assert rec["estimated_cost"] == 50.0
    assert rec["expected_retention_value"] == 200.0

