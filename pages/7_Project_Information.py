"""Page 7: Project Information & Technical Documentation.

Architecture, methodology, dataset documentation, limitations, and platform specifications.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.ui_components import inject_custom_css, render_footer, render_sidebar_filters

st.set_page_config(page_title="Project Information | Churn Intelligence", page_icon="ℹ️", layout="wide")
inject_custom_css()
render_sidebar_filters()

st.title("ℹ️ Project Information & Architecture")
st.markdown("Comprehensive technical documentation, methodology disclosures, system architecture, and machine learning specifications.")

# Section 1: Executive Overview
st.subheader("1. Problem Statement & Motivation")
st.markdown("""
Customer churn represents the primary drain on recurring revenue in subscription-based telecommunications and SaaS business models. Acquiring a replacement customer typically costs **5 to 7 times more** than retaining an existing account.

This platform bridges the gap between raw predictive modeling and frontline business execution:
- **Predict**: Quantify calibrated churn-risk probabilities using cross-validated tree ensembles.
- **Explain**: Deconstruct individual and global churn drivers using **SHAP (SHapley Additive exPlanations)**.
- **Act**: Recommend targeted, positive-EV retention interventions (pricing adjustments, technical support, contract incentives) before churn occurs.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Architecture Diagram
st.subheader("2. System Architecture & End-to-End Pipeline")

mermaid_code = """
flowchart TD
    subgraph Data Layer
        A[IBM Telco Benchmark / Uploaded CSV] --> B[Data Validation & Integrity Checks]
        B --> C[Feature Engineering & Ratio Derivations]
    end

    subgraph Leak-Free Training & Model Selection
        C --> D[Stratified 80/20 Train/Test Split]
        D --> E[Unified Sklearn Pipeline: ColumnTransformer + Model]
        E --> F1[Logistic Regression Benchmark]
        E --> F2[Random Forest + RandomizedSearchCV]
        E --> F3[XGBoost + RandomizedSearchCV]
        F1 & F2 & F3 --> G[Stratified 5-Fold Cross-Validation Selection]
        G --> H[Probability Calibration Comparison: Uncalibrated vs Sigmoid vs Isotonic]
        H --> I[Threshold Optimization on Train Split Only]
        I --> J[One-Time Final Held-Out Test Evaluation + Bootstrap 95% CIs]
    end

    subgraph Explainability & Decision Support
        J --> K[SHAP TreeExplainer & Kernel Fallback]
        K --> L1[Global Feature Importance]
        K --> L2[Local Individual Driver Attribution]
        L2 & J --> M[Intelligent Retention Recommender Engine]
    end

    subgraph Application & Delivery
        M --> N[Streamlit Multi-Page Enterprise Dashboard]
        N --> O1[Executive Overview & Portfolio Exposure]
        N --> O2[Customer Risk Explorer]
        N --> O3[Individual Dossier & SHAP]
        N --> O4[Retention Action Center]
        N --> O5[Model Performance & Calibration]
        N --> O6[Batch CSV Prediction]
    end
"""

st.markdown("```mermaid\n" + mermaid_code + "\n```")

st.markdown("<br>", unsafe_allow_html=True)

# Section 3: Dataset & Feature Engineering
st.subheader("3. Dataset & Feature Engineering")
st.markdown("""
- **Dataset Source**: IBM Telco Customer Churn benchmark (7,043 customer accounts, 21 attributes).
- **Class Imbalance**: 26.5% Churned (`Yes`), 73.5% Retained (`No`).
- **Engineered Features**:
  - `tenure_group`: Discretized lifecycle brackets (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `>60m`).
  - `num_services`: Sum of active subscribed services across voice, internet, streaming, and security.
  - `charges_ratio`: Ratio of total charges billed to expected tenure charges to flag billing anomalies.
  - `monthly_to_total_ratio`: Relative financial velocity indicator.
  - `has_tech_support_or_security`: Key mitigating factor against early cancellation.
  - `cost_per_service`: Average monthly spend per active service.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Machine Learning Protocol
st.subheader("4. Machine Learning & Evaluation Protocol")
st.markdown("""
All models are evaluated on a **strict 80/20 stratified train/test split (N_train = 5,634, N_test = 1,409)**:
- **Zero Leakage**: All numerical standard scalers, categorical one-hot encoders, and hyperparameter tuners are contained in `sklearn.pipeline.Pipeline` objects fit strictly on training splits.
- **5-Fold Stratified Cross-Validation**: Champion model selection (XGBoost, CV ROC-AUC: 0.8487 ± 0.0110) and decision thresholds are determined strictly on training CV.
- **Held-Out Test Set**: Evaluated exactly once for final reporting with non-parametric bootstrap 95% confidence intervals (XGBoost Test ROC-AUC: 0.8449 [0.8224, 0.8653], PR-AUC: 0.6550 [0.6019, 0.7049]).
- **Calibration**: Tested uncalibrated vs. sigmoid vs. isotonic calibration via `CalibratedClassifierCV` using the Brier score.
- **Fairness & Bias Auditing**: Evaluated demographic parity across gender and SeniorCitizen status, plus gender feature ablation (confirming no statistical penalty when omitting gender).
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 5: Methodological Limitations & Future Scope
st.subheader("5. Methodological Limitations & Future Scope")
st.markdown(r"""
1. **Temporal Horizon Absence**: The public IBM Telco dataset is a static cross-sectional snapshot without event timestamps. True longitudinal 30-day forward forecasting requires rolling monthly snapshot logs. The current model produces a calibrated **churn-risk probability estimate**.
2. **Causation vs Statistical Association**: SHAP attributions reflect mathematical feature weights in the model, not verified counterfactual causality. Changing a feature (e.g., offering a discount) may not change the underlying churn probability if the driver is a proxy for unobserved dissatisfaction. Retention actions should be validated via randomized controlled trials (A/B testing).
3. **Illustrative Retention Economics**: Default intervention costs (\$25) and customer acceptance rates (40%) are illustrative defaults configurable via the sidebar. In production, these parameters should be tied to CRM unit economics.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 6: Credits & Authorship
st.subheader("6. Technical Specifications & Repository")
st.markdown("""
- **Project**: Customer Churn Prediction & Intelligent Retention Recommendation System
- **Repository**: [GitHub: SriSaiPranav07/customer-churn-intelligence](https://github.com/SriSaiPranav07/customer-churn-intelligence)
- **Deployment**: Streamlit Community Cloud (Python 3.11)
- **Author**: Sri Sai Pranav ([@SriSaiPranav07](https://github.com/SriSaiPranav07))
""")

render_footer()
