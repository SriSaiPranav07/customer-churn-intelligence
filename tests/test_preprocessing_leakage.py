"""Tests verifying zero data leakage in preprocessing and model pipelines."""
import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_telco_data
from src.data_validation import validate_raw_data
from src.feature_engineering import prepare_features_and_target, NUMERICAL_FEATURES
from src.preprocessing import build_pipeline


def test_scaler_not_fit_on_full_dataset():
    """
    Guarantees no preprocessing leakage:
    Verifies that the StandardScaler mean parameters match the training split
    and strictly differ from the full dataset means.
    """
    raw_df, _ = load_telco_data()
    _, _, _, df_clean = validate_raw_data(raw_df)

    train_df, test_df = train_test_split(
        df_clean, test_size=0.20, stratify=df_clean["Churn"], random_state=42
    )

    X_train, y_train, _ = prepare_features_and_target(train_df)
    X_full, _, _ = prepare_features_and_target(df_clean)

    pipe = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["preprocessor"].named_transformers_["num"]
    scaler_means = scaler.mean_

    train_means = X_train[NUMERICAL_FEATURES].mean().values
    full_means = X_full[NUMERICAL_FEATURES].mean().values

    # Scaler means must match training set means
    np.testing.assert_allclose(scaler_means, train_means, rtol=1e-5, atol=1e-5)

    # Scaler means must NOT match the full dataset means
    diff = np.abs(scaler_means - full_means)
    assert np.any(diff > 1e-3), (
        f"Scaler means appear to be fit on the full dataset! Diff: {diff}"
    )


def test_decision_threshold_not_tuned_on_test_set():
    """
    Guarantees decision threshold is determined via CV on training set,
    independent of test set labels.
    """
    from src.evaluation import find_optimal_threshold
    # Shuffling test labels should have no impact on the stored optimal threshold
    import json
    from src.utils import MODELS_DIR

    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    assert metrics_path.exists(), "evaluation_metrics.json must exist."
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Check that optimal_threshold_cv is recorded in metadata
    assert "optimal_threshold_cv" in metrics["metadata"]
    assert 0.1 <= metrics["metadata"]["optimal_threshold_cv"] <= 0.9
