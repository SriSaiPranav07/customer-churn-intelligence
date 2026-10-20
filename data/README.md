# Dataset Documentation: IBM Telco Customer Churn

## 1. Overview and Source
This project utilizes the benchmark **IBM Telco Customer Churn** dataset.
- **Provider**: IBM Watson Analytics / Kaggle Open Data
- **Format**: CSV format, 7,043 customer accounts
- **Features**: 21 columns capturing demographic data, subscribed products/services, account billing details, and churn status.

## 2. Target Definition
- **Target Column**: `Churn` (`"Yes"` or `"No"`)
- **Definition**: Indicates whether the customer discontinued their service in the historical observation period.
- **Class Distribution**:
  - `No` (Retained): ~73.5% (5,174 records)
  - `Yes` (Churned): ~26.5% (1,869 records)
  - Demonstrates moderate class imbalance (~3:1 retention-to-churn ratio).

## 3. Key Feature Groups
1. **Customer Demographics**:
   - `gender` (Male, Female)
   - `SeniorCitizen` (0 = No, 1 = Yes)
   - `Partner` (Yes, No)
   - `Dependents` (Yes, No)
2. **Account & Contract Attributes**:
   - `tenure` (Number of months the customer has been with the company)
   - `Contract` (Month-to-month, One year, Two year)
   - `PaperlessBilling` (Yes, No)
   - `PaymentMethod` (Electronic check, Mailed check, Bank transfer, Credit card)
   - `MonthlyCharges` (Current monthly fee in USD)
   - `TotalCharges` (Cumulative charges in USD; requires conversion from string and handling for 11 blank entries)
3. **Services Subscribed**:
   - `PhoneService`, `MultipleLines`
   - `InternetService` (DSL, Fiber optic, No)
   - Value-added services: `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

## 4. Methodological Limitations & Prediction Horizon
- **Absence of Longitudinal Timestamps**: The public IBM Telco dataset contains single-point snapshot records with a static binary churn flag. It does not provide event logs or explicit temporal timestamps.
- **Prediction Horizon Disclosure**: Because no future longitudinal labels exist, the predictive model calculates an **estimated churn-risk probability** rather than an empirically validated future 30-day forecast. To establish true 30-day forecasting accuracy in production, longitudinal customer snapshots with rolling 30-day forward outcome windows are required.

## 5. Demonstration & Fallback Data
- When network connectivity is absent, a built-in realistic synthetic generator (`src/data_loader.py:generate_realistic_synthetic_telco_data`) creates realistic data following the identical schema and correlation structure.
- Synthetic records are clearly tagged with `SYNTH-` prefixes and documented metadata so users and reviewers can distinguish synthetic data from historical real data.
