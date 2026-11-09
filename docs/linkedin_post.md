# LinkedIn Post Draft

---

🚀 Just deployed a full **Customer Churn Intelligence & Retention System** — end-to-end, production-style, and open source.

**What it does:**
- Predicts churn probability for every customer using XGBoost (ROC-AUC 0.8449)
- Explains *why* each customer is at risk with SHAP waterfall charts
- Recommends tailored retention actions with expected value ($) calculations
- Runs a full demographic fairness audit across gender and senior citizen segments

**Engineering highlights:**
✅ Zero data leakage — unified sklearn Pipeline; scaler fit on training split only  
✅ 5-fold stratified CV for model selection; held-out test set evaluated exactly once  
✅ Bootstrap 95% confidence intervals for all reported metrics  
✅ Probability calibration (isotonic regression, Brier score optimized)  
✅ Profit curve optimization — finds the threshold that maximizes net retention value  
✅ 17 pytest tests, GitHub Actions CI, ruff linting  

**Tech stack:** Python 3.11 · XGBoost · Scikit-learn · SHAP · Streamlit · Plotly · Joblib

🔗 Live demo: [ADD URL]  
📂 GitHub: https://github.com/SriSaiPranav07/customer-churn-intelligence

#MachineLearning #MLOps #Python #DataScience #XGBoost #SHAP #Streamlit #ChurnPrediction
