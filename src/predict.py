"""Inference engine for customer churn prediction."""
from datetime import datetime, timezone
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Tuple, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd

from src.data_validation import validate_raw_data, validate_uploaded_csv
from src.feature_engineering import prepare_features_and_target
from src.preprocessing import load_preprocessor
from src.utils import (
    MODELS_DIR, get_risk_category,
    RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, logger
)

CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"


class ChurnPredictor:
    """Production inference engine for churn probability and risk categorization."""

    def __init__(self, model_path: Path = CHAMPION_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model and preprocessing artifacts from disk with automatic training fallback."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.preprocessor, self.feature_names = load_preprocessor()
                if self.is_ready():
                    logger.info(f"Loaded champion model and preprocessor ({len(self.feature_names)} features).")
                    return
        except Exception as e:
            logger.warning(f"Could not load pre-existing artifacts: {e}")
            self.model = None
            self.preprocessor = None

        # Self-healing fallback: train on the fly if missing or incompatible
        logger.info("Initializing automated model training fallback...")
        try:
            from src.train import train_and_evaluate_all_models
            train_and_evaluate_all_models()
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.preprocessor, self.feature_names = load_preprocessor()
                logger.info(f"Auto-training completed successfully. Model loaded ({len(self.feature_names) if self.feature_names else 0} features).")
        except Exception as err:
            logger.error(f"Auto-training fallback failed: {err}")

    def is_ready(self) -> bool:
        """Returns True if model and preprocessor are loaded."""
        return self.model is not None and self.preprocessor is not None

    def predict(
        self,
        df: pd.DataFrame,
        high_threshold: float = RISK_THRESHOLD_HIGH,
        med_threshold: float = RISK_THRESHOLD_MEDIUM
    ) -> pd.DataFrame:
        """
        Executes end-to-end churn prediction on a customer DataFrame.
        Returns input dataframe enriched with:
            - churn_probability: float [0, 1]
            - risk_category: "High Risk" | "Medium Risk" | "Low Risk"
            - prediction_timestamp: ISO 8601 string
        """
        if not self.is_ready():
            self._load_artifacts()
            if not self.is_ready():
                raise RuntimeError("ChurnPredictor artifacts could not be initialized.")

        # 1. Validate
        is_valid, errors, warnings, cleaned_df = validate_raw_data(df, is_training=False)
        if not is_valid:
            raise ValueError(f"Input data failed validation: {'; '.join(errors)}")

        # 2. Feature engineering
        X, _, customer_ids = prepare_features_and_target(cleaned_df)

        # 3. Transform
        X_proc = self.preprocessor.transform(X)

        # 4. Predict probabilities
        probabilities = self.model.predict_proba(X_proc)[:, 1]

        # 5. Build output DataFrame
        result_df = cleaned_df.copy()
        result_df["churn_probability"] = np.round(probabilities, 4)
        result_df["risk_category"] = [
            get_risk_category(p, high_thresh=high_threshold, med_thresh=med_threshold)
            for p in probabilities
        ]
        result_df["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

        return result_df


_global_predictor: Optional[ChurnPredictor] = None


def get_predictor() -> ChurnPredictor:
    """Singleton getter for ChurnPredictor."""
    global _global_predictor
    if _global_predictor is None or not _global_predictor.is_ready():
        _global_predictor = ChurnPredictor()
    return _global_predictor
