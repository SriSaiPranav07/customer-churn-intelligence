# Customer Churn Intelligence & Retention System

[![CI](https://github.com/SriSaiPranav07/customer-churn-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/SriSaiPranav07/customer-churn-intelligence/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** `<ADD_LIVE_URL>`

A production-style machine learning platform that predicts customer churn probability, explains the drivers with SHAP, and recommends targeted retention interventions with expected-value calculations — all built on the IBM Telco Customer Churn dataset.

---

## Screenshots

> *[Add screenshots here after deployment]*

---

## Features

| Capability | Implementation |
|---|---|
| Churn probability scoring | XGBoost (CV-selected champion) |
| Explainability | SHAP TreeExplainer, waterfall + global importance |
| Retention recommendations | Rule-based engine with configurable EV model |
| Model comparison | Logistic Regression · Random Forest · XGBoost |
| Probability calibration | Isotonic regression via `CalibratedClassifierCV` |
| Fairness audit | Gender & SeniorCitizen demographic parity |
| Uncertainty quantification | Bootstrap 95% CIs on test ROC-AUC & PR-AUC |
| Data upload | Batch CSV scoring with schema validation |

---

## Model Performance (Real Numbers — No Fabrications)

All metrics are from a **single held-out 20% test split (N = 1,409)**. Preprocessing and model selection were performed strictly on the training split to guarantee zero data leakage.

### Cross-Validation Results (5-Fold Stratified, N_train = 5,634)

| Model | CV ROC-AUC (Mean ± Std) | CV PR-AUC |
|---|---|---|
| Logistic Regression | 0.8457 ± 0.0109 | 0.6608 ± 0.0135 |
| Random Forest | 0.8470 ± 0.0099 | 0.6597 ± 0.0160 |
| **XGBoost (Champion)** | **0.8487 ± 0.0110** | **0.6647 ± 0.0180** |

> XGBoost and Random Forest are statistically comparable within 1 std margin. XGBoost selected as champion based on superior PR-AUC and calibration.

### Held-Out Test Set Results (Evaluated Once)

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier Score |
|---|---|---|---|
| Logistic Regression | 0.8424 [0.8206, 0.8627] | 0.6342 [0.5814, 0.6839] | 0.1668 |
| Random Forest | 0.8416 [0.8191, 0.8623] | 0.6489 [0.5958, 0.6967] | 0.1634 |
| **XGBoost (Champion)** | **0.8449 [0.8224, 0.8653]** | **0.6550 [0.6019, 0.7049]** | **0.1626** |

### Calibration (Champion XGBoost)

| Method | Brier Score |
|---|---|
| Uncalibrated | 0.1626 |
| Sigmoid | 0.1367 |
| **Isotonic (selected)** | **0.1364** |

### Fairness Audit

| Attribute | Subgroup | N | Churn Rate | Recall | Precision | ROC-AUC |
|---|---|---|---|---|---|---|
| Gender | Female | 687 | 28.1% | 47.7% | 68.2% | 0.8436 |
| Gender | Male | 722 | 25.1% | 48.6% | 66.2% | 0.8472 |
| SeniorCitizen | No (0) | 1187 | 23.3% | 45.7% | 66.0% | 0.8456 |
| SeniorCitizen | Yes (1) | 222 | 44.1% | 55.1% | 70.1% | 0.7904 |

**Gender ablation:** Dropping gender as a feature changes test ROC-AUC by +0.0005 (negligible). The model does not rely on gender for predictive power.

---

## Decision Thresholds

| Threshold | Basis | Purpose |
|---|---|---|
| **0.54** (optimal F1) | 5-fold CV on training split only | Default classifier cutoff |
| **0.11** (profit-maximizing) | Retention economics on training CV | Maximum expected net retention value |

---

## Quick Start

```bash
git clone https://github.com/SriSaiPranav07/customer-churn-intelligence.git
cd customer-churn-intelligence
pip install -r requirements.txt
streamlit run app.py
```

> **Note:** Pre-trained model artifacts (`models/champion_model.joblib`, etc.) are committed to the repo. To retrain from scratch:
> ```bash
> python src/train.py
> ```
> Training takes ~3–5 minutes on a standard laptop (8 GB RAM).

---

## Project Structure

```
├── app.py                          # Main Streamlit entry point
├── pages/
│   ├── 1_Executive_Overview.py     # Portfolio KPIs & risk cohorts
│   ├── 2_Customer_Risk_Explorer.py # Filtered roster view
│   ├── 3_Customer_Analysis.py      # Individual dossier + SHAP + EV recommendation
│   ├── 4_Retention_Action_Center.py # Prioritized intervention queue
│   ├── 5_Model_Performance.py      # ROC/PR curves, calibration, fairness, profit curve
│   ├── 6_Data_Upload_Prediction.py # Batch CSV upload & scoring
│   └── 7_Project_Information.py    # Architecture & methodology docs
├── src/
│   ├── data_loader.py              # IBM Telco data acquisition
│   ├── data_validation.py          # Schema & range validation (Pandas)
│   ├── feature_engineering.py      # Reusable feature pipeline
│   ├── preprocessing.py            # ColumnTransformer + Pipeline construction
│   ├── train.py                    # Leakage-free training, 5-fold CV, calibration, fairness
│   ├── evaluation.py               # Metrics, bootstrap CIs, ROC/PR/calibration/profit plots
│   ├── predict.py                  # ChurnPredictor inference engine (public .load() method)
│   ├── explainability.py           # SHAP TreeExplainer with Pipeline/CalibratedCV support
│   ├── retention_recommender.py    # Rule-based recommender with EV = P×acc×ARR - cost
│   ├── ui_components.py            # Sidebar, metric cards, CSS injection
│   └── utils.py                    # Constants, helpers, configurable retention economics
├── models/                         # Trained artifacts (committed for Streamlit Cloud)
│   ├── champion_model.joblib
│   ├── evaluation_metrics.json
│   └── test_evaluation_cache.joblib
├── tests/                          # 17 pytest tests
├── docs/                           # Demo scripts & social post drafts
├── .github/workflows/ci.yml        # GitHub Actions: ruff + pytest
├── pyproject.toml                  # Build system + ruff + pytest config
├── requirements.txt                # Pinned dependencies (Python 3.11)
└── runtime.txt                     # python-3.11 (Streamlit Cloud)
```

---

## ML Engineering Decisions

### Zero Data Leakage
All preprocessing (StandardScaler, OneHotEncoder) lives inside a `sklearn.pipeline.Pipeline` object that is fit strictly on the training split. No scaler or encoder ever sees the test set during fitting.

### Model Selection Protocol
Champion model and decision threshold are selected via 5-fold Stratified CV on the training split. The held-out test set is evaluated exactly once and only for final reporting.

### Probability Calibration
Raw XGBoost scores are distorted near the probability extremes. We compare uncalibrated vs. sigmoid vs. isotonic calibration using `CalibratedClassifierCV` and select the method with the lowest Brier score (isotonic, 0.1364).

### Uncertainty Quantification
Bootstrap 95% confidence intervals (N=1,000 resamples) are reported for all test ROC-AUC and PR-AUC values to communicate statistical uncertainty.

### Profit Curve Optimization
Expected net retention value = `TP_count × gain - flagged_count × cost`. Sweeping thresholds identifies the profit-maximizing cutoff (0.11) for operational deployment.

---

## Limitations

1. **No temporal dimension:** The IBM Telco dataset is a cross-sectional snapshot without event timestamps. Predictions reflect churn propensity estimates, not guaranteed 30-day forecasts.
2. **SHAP is associative, not causal:** SHAP values quantify statistical feature contributions to the model's decision. They do not prove that changing a feature will change churn outcomes. Validate interventions via A/B testing.
3. **Illustrative retention economics:** Default intervention cost (\$25) and acceptance rate (40%) are configurable defaults. In production, tie these to CRM-measured unit economics.

---

## Development

```bash
# Install in editable mode
pip install -e .

# Run all tests
pytest tests -v

# Lint
ruff check .
```

---

## Environment

| Package | Version |
|---|---|
| Python | 3.11.9 |
| scikit-learn | 1.5.2 |
| xgboost | 3.2.0 |
| shap | 0.51.0 |
| pandas | 2.2.3 |
| numpy | 2.1.2 |
| streamlit | ≥1.35 |

---

## Repository

**GitHub:** [SriSaiPranav07/customer-churn-intelligence](https://github.com/SriSaiPranav07/customer-churn-intelligence)  
**Author:** Sri Sai Pranav ([@SriSaiPranav07](https://github.com/SriSaiPranav07))
