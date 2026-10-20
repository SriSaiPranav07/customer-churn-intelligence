"""Model training, hyperparameter tuning, model comparison, and serialization."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data_loader import load_telco_data
from src.data_validation import validate_raw_data
from src.feature_engineering import prepare_features_and_target
from src.preprocessing import fit_and_save_preprocessor
from src.evaluation import compute_comprehensive_metrics, find_optimal_threshold
from src.utils import MODELS_DIR, logger

METRICS_FILE = MODELS_DIR / "evaluation_metrics.json"
CACHE_FILE = MODELS_DIR / "test_evaluation_cache.joblib"


def train_and_evaluate_all_models(
    random_state: int = 42,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Executes the full machine learning training pipeline:
    1. Loads dataset
    2. Validates schema and data ranges
    3. Engineers predictive features
    4. Performs stratified 80/20 train/test split
    5. Fits and persists preprocessing pipeline
    6. Trains Logistic Regression, Random Forest, and XGBoost
    7. Evaluates each model on held-out test data
    8. Selects Champion Model and serializes all artifacts
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initiating model training workflow...")

    # 1. Load Data
    raw_df, meta = load_telco_data(force_download=force_download)
    logger.info(f"Loaded dataset: {len(raw_df)} records. Source: {meta['source']}")

    # 2. Validation
    is_valid, errors, warnings, df_clean = validate_raw_data(raw_df, is_training=True)
    if not is_valid:
        raise ValueError(f"Data validation failed with errors: {errors}")
    for w in warnings:
        logger.warning(w)

    # 3. Feature engineering
    X, y, customer_ids = prepare_features_and_target(df_clean)
    if y is None:
        raise ValueError("Target variable 'Churn' not found in dataset.")

    # 4. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.20, stratify=y, random_state=random_state
    )
    logger.info(f"Training set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # 5. Preprocessing
    preprocessor, feature_names = fit_and_save_preprocessor(X_train)
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Class balance statistics
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    churn_rate = float(np.mean(y == 1))
    logger.info(f"Class imbalance: Retained={n_neg}, Churned={n_pos}, Scale Pos Weight={scale_pos_weight:.2f}")

    models = {}
    evaluation_results = {}
    cache_data = {
        "y_test": y_test.values,
        "customer_ids_test": ids_test.values,
        "feature_names": feature_names,
        "predictions": {}
    }

    # Model A: Logistic Regression Baseline
    logger.info("Training Model A: Logistic Regression Baseline...")
    lr_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    lr_model.fit(X_train_proc, y_train)
    lr_proba = lr_model.predict_proba(X_test_proc)[:, 1]
    lr_thresh, _ = find_optimal_threshold(y_test.values, lr_proba)
    lr_metrics = compute_comprehensive_metrics(y_test.values, lr_proba, threshold=0.5)
    lr_metrics_opt = compute_comprehensive_metrics(y_test.values, lr_proba, threshold=lr_thresh)

    models["Logistic Regression"] = lr_model
    evaluation_results["Logistic Regression"] = {
        "description": "Linear baseline benchmark with balanced class weighting",
        "default_metrics": lr_metrics,
        "optimal_threshold": lr_thresh,
        "optimal_metrics": lr_metrics_opt
    }
    cache_data["predictions"]["Logistic Regression"] = lr_proba
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib", compress=3)

    # Model B: Random Forest Classifier
    logger.info("Training Model B: Random Forest Classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=6,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    rf_model.fit(X_train_proc, y_train)
    rf_proba = rf_model.predict_proba(X_test_proc)[:, 1]
    rf_thresh, _ = find_optimal_threshold(y_test.values, rf_proba)
    rf_metrics = compute_comprehensive_metrics(y_test.values, rf_proba, threshold=0.5)
    rf_metrics_opt = compute_comprehensive_metrics(y_test.values, rf_proba, threshold=rf_thresh)

    models["Random Forest"] = rf_model
    evaluation_results["Random Forest"] = {
        "description": "Ensemble of balanced decision trees capturing non-linear interactions",
        "default_metrics": rf_metrics,
        "optimal_threshold": rf_thresh,
        "optimal_metrics": rf_metrics_opt
    }
    cache_data["predictions"]["Random Forest"] = rf_proba
    joblib.dump(rf_model, MODELS_DIR / "random_forest.joblib", compress=3)

    # Model C: XGBoost Classifier
    if XGBOOST_AVAILABLE:
        logger.info("Training Model C: XGBoost Classifier...")
        xgb_model = XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state
        )
        xgb_model.fit(X_train_proc, y_train)
        xgb_proba = xgb_model.predict_proba(X_test_proc)[:, 1]
        xgb_thresh, _ = find_optimal_threshold(y_test.values, xgb_proba)
        xgb_metrics = compute_comprehensive_metrics(y_test.values, xgb_proba, threshold=0.5)
        xgb_metrics_opt = compute_comprehensive_metrics(y_test.values, xgb_proba, threshold=xgb_thresh)

        models["XGBoost"] = xgb_model
        evaluation_results["XGBoost"] = {
            "description": "Gradient boosted decision trees with regularized learning",
            "default_metrics": xgb_metrics,
            "optimal_threshold": xgb_thresh,
            "optimal_metrics": xgb_metrics_opt
        }
        cache_data["predictions"]["XGBoost"] = xgb_proba
        joblib.dump(xgb_model, MODELS_DIR / "xgboost.joblib", compress=3)
    else:
        logger.warning("XGBoost library unavailable. Relying on Random Forest as primary tree model.")

    # Select Champion Model based on ROC-AUC and F1
    candidates = ["XGBoost", "Random Forest"] if "XGBoost" in models else ["Random Forest"]
    champion_name = max(
        candidates,
        key=lambda m: (evaluation_results[m]["default_metrics"]["roc_auc"] + evaluation_results[m]["default_metrics"]["f1"])
    )
    champion_model = models[champion_name]
    logger.info(f"Champion Model selected: '{champion_name}' with ROC-AUC: {evaluation_results[champion_name]['default_metrics']['roc_auc']}")

    joblib.dump(champion_model, MODELS_DIR / "champion_model.joblib", compress=3)

    # Construct final training summary
    summary_data = {
        "metadata": {
            "dataset_source": meta["source"],
            "is_synthetic": meta["is_synthetic"],
            "total_records": len(df_clean),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "churn_rate": round(churn_rate, 4),
            "scale_pos_weight": round(scale_pos_weight, 2),
            "champion_model_name": champion_name
        },
        "models": evaluation_results,
        "feature_names": feature_names
    }

    # Save metrics JSON
    with open(METRICS_FILE, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save test cache for interactive Streamlit visuals
    joblib.dump(cache_data, CACHE_FILE)
    logger.info("Training pipeline completed successfully. Artifacts saved.")

    return summary_data


if __name__ == "__main__":
    train_and_evaluate_all_models()
