# Model Artifacts and Evaluation Documentation

This directory stores the trained machine learning pipelines, evaluation metrics, and preprocessor artifacts.

## Serialized Artifacts
1. **`preprocessor.joblib`**: Fitted Scikit-learn `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder(handle_unknown='ignore')` for categorical features. Contains 67 encoded input feature names.
2. **`champion_model.joblib`**: The production champion model (Random Forest / XGBoost) selected based on held-out ROC-AUC and F1 performance.
3. **`logistic_regression.joblib`**: Linear baseline benchmark model.
4. **`random_forest.joblib`**: Balanced random forest ensemble.
5. **`xgboost.joblib`**: Gradient boosted trees model.
6. **`evaluation_metrics.json`**: Exact, un-fabricated test evaluation metrics across all models, optimal decision thresholds, and confusion matrix cell counts.
7. **`test_evaluation_cache.joblib`**: Cached predictions on the held-out 1,409 test records for instant, interactive ROC and PR curve rendering in Streamlit.

## Training & Retraining
To reproduce training and update model artifacts from command line:
```bash
python -m src.train
```
Or execute through the Streamlit web dashboard.
