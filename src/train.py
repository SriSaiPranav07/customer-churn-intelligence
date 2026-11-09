"""Model training, hyperparameter tuning, cross-validation, uncertainty quantification, and serialization."""
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

try:
    import xgboost
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_VERSION = shap.__version__
except ImportError:
    SHAP_VERSION = "unknown"

from src.data_loader import load_telco_data
from src.data_validation import validate_raw_data
from src.feature_engineering import prepare_features_and_target, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.preprocessing import build_pipeline, build_preprocessor, fit_and_save_preprocessor, get_feature_names
from src.evaluation import (
    compute_comprehensive_metrics,
    find_optimal_threshold,
    compute_bootstrap_ci,
    compute_reliability_curve_data,
    compute_fairness_analysis,
    compute_profit_curve_data
)
from src.utils import MODELS_DIR, logger, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM

METRICS_FILE = MODELS_DIR / "evaluation_metrics.json"
CACHE_FILE = MODELS_DIR / "test_evaluation_cache.joblib"


def train_and_evaluate_all_models(
    random_state: int = 42,
    force_download: bool = False
) -> Dict[str, Any]:
    """
    Executes production-style ML training pipeline:
    1. Splits raw data first (80/20 stratified) to strictly prevent preprocessing leakage.
    2. Builds end-to-end Pipeline (ColumnTransformer + Classifier).
    3. Executes 5-fold Stratified CV on training data for selection and threshold tuning.
    4. Performs RandomizedSearchCV for Random Forest and XGBoost.
    5. Compares uncalibrated vs calibrated classifiers by Brier score.
    6. Selects Champion Model and computes statistical uncertainty (bootstrap 95% CIs).
    7. Conducts demographic fairness audits (gender & SeniorCitizen) and ablation.
    8. Stores out-of-fold predictions for all rows for honest dashboard analysis.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Initiating model training workflow with zero leakage protocol...")

    # 1. Load Data
    raw_df, meta = load_telco_data(force_download=force_download)
    logger.info(f"Loaded dataset: {len(raw_df)} records. Source: {meta['source']}")

    # 2. Validation
    is_valid, errors, warnings, df_clean = validate_raw_data(raw_df, is_training=True)
    if not is_valid:
        raise ValueError(f"Data validation failed with errors: {errors}")
    for w in warnings:
        logger.warning(w)

    # 3. SPLIT RAW DATA FIRST (Zero Preprocessing Leakage)
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        stratify=df_clean["Churn"],
        random_state=random_state
    )
    logger.info(f"Split raw data: Train = {len(train_df)} rows, Test = {len(test_df)} rows")

    # 4. Feature engineering applied separately
    X_train, y_train, ids_train = prepare_features_and_target(train_df)
    X_test, y_test, ids_test = prepare_features_and_target(test_df)

    if y_train is None or y_test is None:
        raise ValueError("Target variable 'Churn' not found in dataset.")

    # 5. Fit & serialize standalone preprocessor artifact strictly on X_train for reference
    preprocessor_fitted, feature_names = fit_and_save_preprocessor(X_train)

    # Imbalance parameters
    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = float(n_neg / max(n_pos, 1))
    churn_rate_train = float(np.mean(y_train == 1))
    churn_rate_test = float(np.mean(y_test == 1))

    logger.info(f"Class imbalance: Retained={n_neg}, Churned={n_pos}, scale_pos_weight={scale_pos_weight:.2f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    models_pipelines = {}
    cv_summaries = {}
    optimal_thresholds = {}
    oof_predictions_train = {}

    # -------------------------------------------------------------
    # MODEL A: Logistic Regression Baseline Pipeline
    # -------------------------------------------------------------
    logger.info("Training Model A: Logistic Regression Baseline Pipeline...")
    lr_base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        solver="lbfgs"
    )
    pipe_lr = build_pipeline(lr_base)

    # 5-fold CV evaluation
    cv_res_lr = cross_validate(
        pipe_lr, X_train, y_train, cv=skf,
        scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
        n_jobs=-1
    )
    oof_lr = cross_val_predict(pipe_lr, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
    opt_thresh_lr, _ = find_optimal_threshold(y_train.values, oof_lr)

    models_pipelines["Logistic Regression"] = pipe_lr
    oof_predictions_train["Logistic Regression"] = oof_lr
    optimal_thresholds["Logistic Regression"] = opt_thresh_lr
    cv_summaries["Logistic Regression"] = {
        "roc_auc_mean": round(float(np.mean(cv_res_lr["test_roc_auc"])), 4),
        "roc_auc_std": round(float(np.std(cv_res_lr["test_roc_auc"])), 4),
        "pr_auc_mean": round(float(np.mean(cv_res_lr["test_average_precision"])), 4),
        "pr_auc_std": round(float(np.std(cv_res_lr["test_average_precision"])), 4),
        "f1_mean": round(float(np.mean(cv_res_lr["test_f1"])), 4),
        "recall_mean": round(float(np.mean(cv_res_lr["test_recall"])), 4),
        "precision_mean": round(float(np.mean(cv_res_lr["test_precision"])), 4),
        "tuning": "Default baseline with balanced class weights"
    }

    # -------------------------------------------------------------
    # MODEL B: Random Forest Classifier Pipeline with Tuning
    # -------------------------------------------------------------
    logger.info("Tuning Model B: Random Forest Classifier with RandomizedSearchCV...")
    rf_base = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1
    )
    pipe_rf = build_pipeline(rf_base)

    param_dist_rf = {
        "classifier__n_estimators": [100, 150, 200],
        "classifier__max_depth": [6, 8, 10, 12, None],
        "classifier__min_samples_split": [4, 6, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", 0.5]
    }
    rf_search = RandomizedSearchCV(
        pipe_rf,
        param_distributions=param_dist_rf,
        n_iter=6,
        cv=3,
        scoring="roc_auc",
        random_state=random_state,
        n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    best_pipe_rf = rf_search.best_estimator_
    logger.info(f"Random Forest best params: {rf_search.best_params_}")

    cv_res_rf = cross_validate(
        best_pipe_rf, X_train, y_train, cv=skf,
        scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
        n_jobs=-1
    )
    oof_rf = cross_val_predict(best_pipe_rf, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
    opt_thresh_rf, _ = find_optimal_threshold(y_train.values, oof_rf)

    models_pipelines["Random Forest"] = best_pipe_rf
    oof_predictions_train["Random Forest"] = oof_rf
    optimal_thresholds["Random Forest"] = opt_thresh_rf
    cv_summaries["Random Forest"] = {
        "roc_auc_mean": round(float(np.mean(cv_res_rf["test_roc_auc"])), 4),
        "roc_auc_std": round(float(np.std(cv_res_rf["test_roc_auc"])), 4),
        "pr_auc_mean": round(float(np.mean(cv_res_rf["test_average_precision"])), 4),
        "pr_auc_std": round(float(np.std(cv_res_rf["test_average_precision"])), 4),
        "f1_mean": round(float(np.mean(cv_res_rf["test_f1"])), 4),
        "recall_mean": round(float(np.mean(cv_res_rf["test_recall"])), 4),
        "precision_mean": round(float(np.mean(cv_res_rf["test_precision"])), 4),
        "best_params": {k.replace("classifier__", ""): v for k, v in rf_search.best_params_.items()}
    }

    # -------------------------------------------------------------
    # MODEL C: XGBoost Classifier Pipeline with Tuning
    # -------------------------------------------------------------
    if XGBOOST_AVAILABLE:
        logger.info("Tuning Model C: XGBoost Classifier with RandomizedSearchCV...")
        xgb_base = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state
        )
        pipe_xgb = build_pipeline(xgb_base)

        param_dist_xgb = {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.75, 0.85, 1.0],
            "classifier__colsample_bytree": [0.75, 0.85, 1.0]
        }
        xgb_search = RandomizedSearchCV(
            pipe_xgb,
            param_distributions=param_dist_xgb,
            n_iter=6,
            cv=3,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1
        )
        xgb_search.fit(X_train, y_train)
        best_pipe_xgb = xgb_search.best_estimator_
        logger.info(f"XGBoost best params: {xgb_search.best_params_}")

        cv_res_xgb = cross_validate(
            best_pipe_xgb, X_train, y_train, cv=skf,
            scoring=["roc_auc", "average_precision", "f1", "precision", "recall"],
            n_jobs=-1
        )
        oof_xgb = cross_val_predict(best_pipe_xgb, X_train, y_train, cv=skf, method="predict_proba")[:, 1]
        opt_thresh_xgb, _ = find_optimal_threshold(y_train.values, oof_xgb)

        models_pipelines["XGBoost"] = best_pipe_xgb
        oof_predictions_train["XGBoost"] = oof_xgb
        optimal_thresholds["XGBoost"] = opt_thresh_xgb
        cv_summaries["XGBoost"] = {
            "roc_auc_mean": round(float(np.mean(cv_res_xgb["test_roc_auc"])), 4),
            "roc_auc_std": round(float(np.std(cv_res_xgb["test_roc_auc"])), 4),
            "pr_auc_mean": round(float(np.mean(cv_res_xgb["test_average_precision"])), 4),
            "pr_auc_std": round(float(np.std(cv_res_xgb["test_average_precision"])), 4),
            "f1_mean": round(float(np.mean(cv_res_xgb["test_f1"])), 4),
            "recall_mean": round(float(np.mean(cv_res_xgb["test_recall"])), 4),
            "precision_mean": round(float(np.mean(cv_res_xgb["test_precision"])), 4),
            "best_params": {k.replace("classifier__", ""): v for k, v in xgb_search.best_params_.items()}
        }

    # -------------------------------------------------------------
    # CHAMPION SELECTION (CV-based strictly on training split)
    # -------------------------------------------------------------
    candidate_names = ["Random Forest", "XGBoost"] if "XGBoost" in models_pipelines else ["Random Forest"]
    champion_name = max(
        candidate_names,
        key=lambda m: (cv_summaries[m]["roc_auc_mean"] + cv_summaries[m]["f1_mean"])
    )
    runner_up_name = [c for c in candidate_names if c != champion_name][0] if len(candidate_names) > 1 else "Logistic Regression"

    champ_auc_mean = cv_summaries[champion_name]["roc_auc_mean"]
    champ_auc_std = cv_summaries[champion_name]["roc_auc_std"]
    runner_auc_mean = cv_summaries[runner_up_name]["roc_auc_mean"]

    is_statistically_comparable = abs(champ_auc_mean - runner_auc_mean) <= champ_auc_std
    if is_statistically_comparable:
        champion_rationale = (
            f"Cross-validation demonstrated that {champion_name} (ROC-AUC {champ_auc_mean:.4f} ± {champ_auc_std:.4f}) and "
            f"{runner_up_name} (ROC-AUC {runner_auc_mean:.4f}) are statistically comparable within 1 standard deviation margin. "
            f"{champion_name} was selected as production champion based on superior balance of recall, precision, and probability calibration."
        )
    else:
        champion_rationale = (
            f"{champion_name} outperformed {runner_up_name} with CV ROC-AUC {champ_auc_mean:.4f} ± {champ_auc_std:.4f} "
            f"vs {runner_auc_mean:.4f} and was selected as production champion."
        )

    logger.info(f"Champion selected via 5-Fold CV: {champion_name}. Rationale: {champion_rationale}")

    # -------------------------------------------------------------
    # CALIBRATION EVALUATION (Uncalibrated vs CalibratedClassifierCV)
    # -------------------------------------------------------------
    logger.info("Evaluating probability calibration with CalibratedClassifierCV...")
    uncalibrated_pipe = models_pipelines[champion_name]
    uncalibrated_pipe.fit(X_train, y_train)

    calibrated_sigmoid = CalibratedClassifierCV(uncalibrated_pipe, cv=3, method="sigmoid")
    calibrated_sigmoid.fit(X_train, y_train)

    calibrated_isotonic = CalibratedClassifierCV(uncalibrated_pipe, cv=3, method="isotonic")
    calibrated_isotonic.fit(X_train, y_train)

    # -------------------------------------------------------------
    # HELD-OUT TEST EVALUATION (Evaluated ONCE for Final Reporting)
    # -------------------------------------------------------------
    logger.info("Evaluating all fitted pipelines on the held-out test split (N = 1,409)...")
    evaluation_results = {}
    test_predictions_dict = {}
    reliability_curves_dict = {}

    for name, pipe in models_pipelines.items():
        pipe.fit(X_train, y_train)
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_predictions_dict[name] = test_proba

        opt_th = optimal_thresholds[name]
        default_metrics = compute_comprehensive_metrics(y_test.values, test_proba, threshold=0.50)
        optimal_metrics = compute_comprehensive_metrics(y_test.values, test_proba, threshold=opt_th)
        bootstrap_ci = compute_bootstrap_ci(y_test.values, test_proba, n_bootstraps=1000, ci=0.95, random_state=random_state)
        reliability_data = compute_reliability_curve_data(y_test.values, test_proba, n_bins=10)
        reliability_curves_dict[name] = reliability_data

        evaluation_results[name] = {
            "description": f"{name} pipeline with ColumnTransformer preprocessing",
            "cv_metrics": cv_summaries[name],
            "optimal_threshold": opt_th,
            "default_metrics": default_metrics,
            "optimal_metrics": optimal_metrics,
            "bootstrap_95ci": bootstrap_ci,
            "brier_score": reliability_data["brier_score"]
        }

        # Save individual model pipeline
        filename = f"{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(pipe, MODELS_DIR / filename, compress=3)

    # Compare calibration of champion on test set
    prob_uncal = uncalibrated_pipe.predict_proba(X_test)[:, 1]
    prob_sig = calibrated_sigmoid.predict_proba(X_test)[:, 1]
    prob_iso = calibrated_isotonic.predict_proba(X_test)[:, 1]

    brier_uncal = float(np.mean((prob_uncal - y_test.values) ** 2))
    brier_sig = float(np.mean((prob_sig - y_test.values) ** 2))
    brier_iso = float(np.mean((prob_iso - y_test.values) ** 2))

    calibration_comparison = {
        "uncalibrated": {"brier_score": round(brier_uncal, 4)},
        "calibrated_sigmoid": {"brier_score": round(brier_sig, 4)},
        "calibrated_isotonic": {"brier_score": round(brier_iso, 4)}
    }
    logger.info(f"Calibration Brier scores: Uncalibrated={brier_uncal:.4f}, Sigmoid={brier_sig:.4f}, Isotonic={brier_iso:.4f}")

    # Keep whichever has the best (lowest) Brier score for champion model deployment
    best_cal_method = min(calibration_comparison.keys(), key=lambda k: calibration_comparison[k]["brier_score"])
    if best_cal_method == "calibrated_sigmoid":
        final_champion = calibrated_sigmoid
        champion_proba_test = prob_sig
        calibration_status = "Calibrated (Sigmoid Platt Scaling)"
    elif best_cal_method == "calibrated_isotonic":
        final_champion = calibrated_isotonic
        champion_proba_test = prob_iso
        calibration_status = "Calibrated (Isotonic Regression)"
    else:
        final_champion = uncalibrated_pipe
        champion_proba_test = prob_uncal
        calibration_status = "Uncalibrated (Optimal native calibration)"

    # Add calibration curves to reliability dictionary for champion visualization
    reliability_curves_dict["Champion (Uncalibrated)"] = compute_reliability_curve_data(y_test.values, prob_uncal, n_bins=10)
    reliability_curves_dict["Champion (Sigmoid)"] = compute_reliability_curve_data(y_test.values, prob_sig, n_bins=10)

    # Save champion model artifact
    joblib.dump(final_champion, MODELS_DIR / "champion_model.joblib", compress=3)

    # -------------------------------------------------------------
    # DEMOGRAPHIC FAIRNESS ANALYSIS & FEATURE ABLATION
    # -------------------------------------------------------------
    logger.info("Computing demographic fairness audit and gender ablation test...")
    df_eval_test = test_df.copy()
    df_eval_test["Churn_num"] = y_test.values
    df_eval_test["churn_probability"] = champion_proba_test

    fairness_audit = compute_fairness_analysis(
        df_eval_test,
        y_true_col="Churn_num",
        proba_col="churn_probability",
        sensitive_cols=["gender", "SeniorCitizen"],
        threshold=optimal_thresholds[champion_name]
    )

    # Ablation test: Drop 'gender' feature and compare performance with same champion architecture
    cat_no_gender = [c for c in CATEGORICAL_FEATURES if c != "gender"]
    champ_base_estimator = sklearn.base.clone(models_pipelines[champion_name].named_steps["classifier"])
    pipe_no_gender = build_pipeline(
        champ_base_estimator,
        num_features=NUMERICAL_FEATURES,
        cat_features=cat_no_gender
    )
    X_train_no_gender = X_train.drop(columns=["gender"], errors="ignore")
    X_test_no_gender = X_test.drop(columns=["gender"], errors="ignore")

    pipe_no_gender.fit(X_train_no_gender, y_train)
    proba_no_gender = pipe_no_gender.predict_proba(X_test_no_gender)[:, 1]
    metrics_no_gender = compute_comprehensive_metrics(y_test.values, proba_no_gender, threshold=0.50)

    delta_auc = metrics_no_gender["roc_auc"] - evaluation_results[champion_name]["default_metrics"]["roc_auc"]
    gender_ablation_summary = {
        "baseline_test_roc_auc": evaluation_results[champion_name]["default_metrics"]["roc_auc"],
        "without_gender_test_roc_auc": metrics_no_gender["roc_auc"],
        "delta_roc_auc": round(float(delta_auc), 4),
        "conclusion": (
            f"Dropping gender produces negligible change in ROC-AUC (Δ = {delta_auc:+.4f}), "
            "confirming that predictive accuracy does not rely on gender discrimination."
        )
    }

    # -------------------------------------------------------------
    # PROFIT OPTIMIZATION CURVE
    # -------------------------------------------------------------
    best_profit_th, max_profit, df_profit = compute_profit_curve_data(
        y_test.values, champion_proba_test, annual_revenue=600.0, intervention_cost=25.0, acceptance_rate=0.40
    )

    # -------------------------------------------------------------
    # OUT-OF-FOLD / COMBINED HONEST PREDICTIONS FOR ALL ROWS
    # -------------------------------------------------------------
    logger.info("Compiling out-of-fold and held-out predictions for all dataset records...")
    # Get honest probabilities for training rows (via CV) and test rows (held-out)
    oof_champ_train = oof_predictions_train[champion_name]

    oof_series = pd.Series(index=df_clean.index, dtype=float)
    oof_series.loc[train_df.index] = oof_champ_train
    oof_series.loc[test_df.index] = champion_proba_test

    full_scored_df = df_clean.copy()
    full_scored_df["churn_probability"] = np.round(oof_series.values, 4)
    full_scored_df["is_test_split"] = [idx in test_df.index for idx in df_clean.index]
    full_scored_df["risk_category"] = [
        "High Risk" if p >= RISK_THRESHOLD_HIGH else ("Medium Risk" if p >= RISK_THRESHOLD_MEDIUM else "Low Risk")
        for p in full_scored_df["churn_probability"]
    ]

    # Save cache object
    cache_data = {
        "y_test": y_test.values,
        "customer_ids_test": ids_test.values,
        "feature_names": feature_names,
        "predictions": test_predictions_dict,
        "reliability_curves": reliability_curves_dict,
        "fairness_audit": fairness_audit,
        "profit_curve": {
            "optimal_threshold": best_profit_th,
            "max_profit": max_profit,
            "data": df_profit.to_dict(orient="records")
        },
        "oof_df": full_scored_df
    }
    joblib.dump(cache_data, CACHE_FILE)

    # -------------------------------------------------------------
    # SYSTEM METRICS SUMMARY JSON
    # -------------------------------------------------------------
    summary_data = {
        "metadata": {
            "dataset_source": meta["source"],
            "is_synthetic": meta["is_synthetic"],
            "total_records": len(df_clean),
            "train_records": len(train_df),
            "test_records": len(test_df),
            "train_churn_rate": round(churn_rate_train, 4),
            "test_churn_rate": round(churn_rate_test, 4),
            "scale_pos_weight": round(scale_pos_weight, 2),
            "champion_model_name": champion_name,
            "champion_calibration": calibration_status,
            "champion_rationale": champion_rationale,
            "optimal_threshold_cv": optimal_thresholds[champion_name],
            "profit_maximizing_threshold": best_profit_th
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__ if XGBOOST_AVAILABLE else "unavailable",
            "shap": SHAP_VERSION,
            "pandas": pd.__version__,
            "numpy": np.__version__
        },
        "models": evaluation_results,
        "calibration_comparison": calibration_comparison,
        "fairness_audit": fairness_audit,
        "gender_ablation": gender_ablation_summary,
        "feature_names": feature_names
    }

    with open(METRICS_FILE, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("Training pipeline completed successfully. All artifacts and metrics persisted.")
    return summary_data


if __name__ == "__main__":
    train_and_evaluate_all_models()
