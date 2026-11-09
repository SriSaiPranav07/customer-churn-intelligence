"""Inference engine for customer churn prediction."""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_validation import validate_raw_data
from src.feature_engineering import prepare_features_and_target
from src.preprocessing import get_feature_names, load_preprocessor
from src.utils import MODELS_DIR, RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM, get_risk_category, logger

CHAMPION_MODEL_PATH = MODELS_DIR / "champion_model.joblib"


class ChurnPredictor:
    """Production inference engine for churn probability and risk categorization."""

    def __init__(self, model_path: Path = CHAMPION_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.load()

    def load(self) -> "ChurnPredictor":
        """Public loader for predictor artifacts."""
        if not self.model_path.exists():
            logger.warning(f"Champion model not found at {self.model_path}")
            return self

        try:
            loaded_artifact = joblib.load(self.model_path)
            self.model = loaded_artifact

            # Extract preprocessor and feature names
            if isinstance(loaded_artifact, Pipeline):
                pre = loaded_artifact.named_steps.get("preprocessor")
                self.preprocessor = pre
                if pre is not None:
                    self.feature_names = get_feature_names(pre)
            elif hasattr(loaded_artifact, "calibrated_classifiers_") and loaded_artifact.calibrated_classifiers_:
                base = loaded_artifact.calibrated_classifiers_[0].estimator
                if isinstance(base, Pipeline):
                    pre = base.named_steps.get("preprocessor")
                    self.preprocessor = pre
                    if pre is not None:
                        self.feature_names = get_feature_names(pre)
            else:
                self.preprocessor, self.feature_names = load_preprocessor()

            feat_count = len(self.feature_names) if self.feature_names else 0
            logger.info(f"Loaded champion model and preprocessor ({feat_count} features).")
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            self.model = None
            self.preprocessor = None
            self.feature_names = None

        return self

    def _model_handles_raw_features(self) -> bool:
        """Determines if the loaded model object internally contains preprocessing."""
        if isinstance(self.model, Pipeline):
            return True
        if hasattr(self.model, "calibrated_classifiers_") and self.model.calibrated_classifiers_:
            if isinstance(self.model.calibrated_classifiers_[0].estimator, Pipeline):
                return True
        return False

    def is_ready(self) -> bool:
        """Returns True if model and preprocessor are loaded."""
        return self.model is not None

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
            raise RuntimeError("ChurnPredictor artifacts are not loaded. Run `python -m src.train` first.")

        # 1. Validate
        is_valid, errors, warnings, cleaned_df = validate_raw_data(df, is_training=False)
        if not is_valid:
            raise ValueError(f"Input data failed validation: {'; '.join(errors)}")

        # 2. Feature engineering
        X, _, _ = prepare_features_and_target(cleaned_df)

        # 3. Predict probabilities
        if self._model_handles_raw_features():
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            if self.preprocessor is None:
                self.preprocessor, self.feature_names = load_preprocessor()
            X_proc = self.preprocessor.transform(X)
            probabilities = self.model.predict_proba(X_proc)[:, 1]

        # 4. Build output DataFrame
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
