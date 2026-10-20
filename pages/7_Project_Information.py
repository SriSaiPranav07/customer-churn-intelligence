"""Page 7: Project Information & Technical Documentation.

Architecture, methodology, dataset documentation, limitations, and Zelvora Industry Labs credits.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from src.ui_components import inject_custom_css, render_sidebar_filters, render_footer

st.set_page_config(page_title="Project Information | Churn Intelligence", page_icon="ℹ️", layout="wide")
inject_custom_css()
render_sidebar_filters()

st.title("ℹ️ Project Information & Architecture")
st.markdown("Comprehensive technical documentation, methodology disclosures, system architecture, and project credits.")

# Section 1: Executive Overview
st.subheader("1. Problem Statement & Motivation")
st.markdown("""
Customer churn represents the primary drain on recurring revenue in subscription-based telecommunications and SaaS business models. Acquiring a replacement customer typically costs **5 to 7 times more** than retaining an existing account.

This platform bridges the gap between raw predictive modeling and frontline business execution:
- **Predict**: Quantify the churn-risk probability for every customer account.
- **Explain**: Deconstruct individual and global churn drivers using **SHAP (SHapley Additive exPlanations)**.
- **Act**: Recommend targeted, high-ROI retention interventions (pricing adjustments, technical support, contract incentives) before churn occurs.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 2: Architecture Diagram
st.subheader("2. System Architecture & End-to-End Pipeline")

mermaid_code = """
flowchart TD
    subgraph Data Layer
        A[IBM Telco Churn CSV / Synthetic Fallback] --> B[Data Validation & Integrity Checks]
        B --> C[Feature Engineering & Leakage Isolation]
    end

    subgraph Modeling & Preprocessing
        C --> D[ColumnTransformer Preprocessor]
        D --> E1[Logistic Regression Benchmark]
        D --> E2[Random Forest Ensemble]
        D --> E3[XGBoost Gradient Boosting]
        E1 & E2 & E3 --> F[Stratified Test Set Evaluation]
        F --> G[Champion Model Selection]
    end

    subgraph Explainability & Decision Support
        G --> H[SHAP TreeExplainer]
        H --> I1[Global Feature Importance]
        H --> I2[Local Individual Driver Attribution]
        I2 & G --> J[Intelligent Retention Recommender Engine]
    end

    subgraph Application & Delivery
        J --> K[Streamlit Multi-Page Dashboard]
        K --> L1[Executive Overview]
        K --> L2[Customer Risk Explorer]
        K --> L3[Individual Dossier & SHAP]
        K --> L4[Retention Action Center]
        K --> L5[Batch CSV Prediction]
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
  - `charges_ratio`: Ratio of total charges billed to expected tenure charges to flag billing rate anomalies.
  - `monthly_to_total_ratio`: Relative financial velocity indicator.
  - `has_tech_support_or_security`: Key mitigating factor against early cancellation.
  - `cost_per_service`: Average monthly spend per active service.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 4: Machine Learning Protocol
st.subheader("4. Machine Learning & Evaluation Protocol")
st.markdown("""
All models are evaluated on a **strict 80/20 stratified train/test split (N_test = 1,409)**:
- **Baseline (Logistic Regression)**: Linear benchmark with `class_weight='balanced'`. Test ROC-AUC = 0.8424.
- **Random Forest**: 150 trees, max depth 10, `class_weight='balanced_subsample'`. Test ROC-AUC = 0.8388, PR-AUC = 0.6452.
- **XGBoost**: 150 estimators, learning rate 0.05, `scale_pos_weight=2.77`. Test ROC-AUC = 0.8428, PR-AUC = 0.6562.
- **Decision Threshold Optimization**: The system sweeps threshold cutoffs to identify optimal operational settings (e.g. threshold = 0.42 captures 78.6% of churners while controlling false alarms).
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 5: Limitations & Future Enhancements
st.subheader("5. Methodological Limitations & Future Scope")
st.markdown("""
1. **Temporal Horizon Absence**: The public IBM Telco dataset is a cross-sectional snapshot without event timestamps. True longitudinal 30-day forward forecasting requires rolling monthly snapshot logs. The current model produces a calibrated **churn-risk probability estimate**.
2. **Causation vs Statistical Influence**: SHAP attributions reflect mathematical feature weights in the model, not verified counterfactual causality. Retention interventions should be validated via A/B testing.
3. **Future Scope**: Integration of survival analysis (Cox Proportional Hazards) for time-to-event modeling and automated Uplift Modeling to distinguish persuadable customers from lost causes.
""")

st.markdown("<br>", unsafe_allow_html=True)

# Section 6: Credits & Authorship
st.subheader("6. Project Credits & Affiliation")
st.markdown("""
- **Project**: Customer Churn Prediction & Intelligent Retention Recommendation System
- **Context**: **Zelvora Technologies Pvt. Ltd. — Zelvora Industry Labs, Artificial Intelligence Week 3: Level Up Project**
- **Repository**: [GitHub: SriSaiPranav07/customer-churn-intelligence](https://github.com/SriSaiPranav07/customer-churn-intelligence)
- **Developer**: Sri Sai Pranav ([@SriSaiPranav07](https://github.com/SriSaiPranav07))
""")

render_footer()
