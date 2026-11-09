# 90-Second Live Demo Script

## Setup (Before Demo)
1. Run `streamlit run app.py`
2. Have the browser open to the homepage
3. Data pre-loaded (7,043 IBM Telco accounts)

---

## Script

### [0:00 – 0:15] Hook
> "Every telecom loses thousands in revenue each month to customer churn. This platform predicts who is about to leave — and tells your team exactly what to do before they do."

### [0:15 – 0:35] Executive Overview
- Navigate to **Executive Overview**
- Point to MRR at Risk and high-risk account count
- Highlight the tenure lifecycle chart: "New customers are 3× more likely to churn"

### [0:35 – 0:55] Customer Dossier & SHAP
- Navigate to **Customer Analysis**
- Select a High Risk customer
- Show the SHAP waterfall: "Month-to-month contract and electronic check are the two biggest drivers"
- Show the recommended action with expected value

### [0:55 – 1:15] Model Performance
- Navigate to **Model Performance**
- Show CV table with confidence intervals: "XGBoost is champion with 0.8449 ROC-AUC"
- Show calibration reliability curve

### [1:15 – 1:30] Close
> "Built with XGBoost, SHAP, and Streamlit. Zero API keys. Deployable in minutes. Try it at the link in the description."

---

## Key Numbers to Cite
- 7,043 customer accounts analyzed
- XGBoost Test ROC-AUC: **0.8449** [95% CI: 0.8224 – 0.8653]
- MRR at risk (varies by threshold, default High: 0.60)
- Zero data leakage — scaler fit on training data only
