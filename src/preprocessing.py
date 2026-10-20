"""Data preprocessing and feature transformation pipelines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Tuple, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import MODELS_DIR, logger

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def build_preprocessor(
    num_features: List[str] = NUMERICAL_FEATURES,
    cat_features: List[str] = CATEGORICAL_FEATURES
) -> ColumnTransformer:
    """Creates a ColumnTransformer pipeline for numeric scaling and one-hot categorical encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ],
        remainder="drop"
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extracts human-readable feature names from the fitted preprocessor."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if trans == "drop" or name == "remainder":
            continue
        if hasattr(trans, "get_feature_names_out"):
            names = trans.get_feature_names_out(cols)
            feature_names.extend(names)
        else:
            feature_names.extend(cols)
    return feature_names


def fit_and_save_preprocessor(
    X_train: pd.DataFrame,
    target_path: Path = PREPROCESSOR_PATH
) -> Tuple[ColumnTransformer, List[str]]:
    """Fits preprocessor on training data and serializes to disk."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    feature_names = get_feature_names(preprocessor)

    joblib.dump({"preprocessor": preprocessor, "feature_names": feature_names}, target_path, compress=3)
    logger.info(f"Fitted preprocessor saved to {target_path} with {len(feature_names)} transformed features.")
    return preprocessor, feature_names


def load_preprocessor(target_path: Path = PREPROCESSOR_PATH) -> Tuple[Optional[ColumnTransformer], Optional[List[str]]]:
    """Loads saved preprocessor artifact from disk."""
    if not target_path.exists():
        logger.warning(f"Preprocessor file not found at {target_path}")
        return None, None
    try:
        data = joblib.load(target_path)
        return data["preprocessor"], data["feature_names"]
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        return None, None
