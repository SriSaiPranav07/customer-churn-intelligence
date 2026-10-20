"""Data loading, acquisition, and synthetic fallback generation."""
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import urllib.request
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

from src.utils import DATA_DIR, logger

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
LOCAL_DATA_FILE = DATA_DIR / "Telco-Customer-Churn.csv"
SAMPLE_UPLOAD_FILE = DATA_DIR / "sample_customers.csv"


def download_ibm_telco_data(target_path: Path = LOCAL_DATA_FILE) -> bool:
    """Download the official IBM Telco Customer Churn dataset."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading IBM Telco dataset from {DATA_URL} to {target_path}...")
    try:
        # User-agent header to avoid bot blockers
        req = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=15) as response, open(target_path, "wb") as out_file:
            out_file.write(response.read())
        logger.info("Dataset downloaded successfully.")
        return True
    except Exception as e:
        logger.warning(f"Download failed with error: {e}. Falling back to synthetic generator.")
        return False


def generate_realistic_synthetic_telco_data(n_samples: int = 7043, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic customer churn dataset following the exact
    IBM Telco schema, distributions, and churn correlations.
    Used strictly as an offline fallback when network acquisition is unavailable.
    """
    logger.info(f"Generating {n_samples} realistic synthetic customer records...")
    rng = np.random.default_rng(random_state)

    customer_ids = [f"SYNTH-{10000 + i}" for i in range(n_samples)]
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(partner == "Yes", rng.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45]), "No")
    
    # Tenure follows a bimodal distribution: many new (<12m) and many long-term (>60m)
    tenure_raw = np.concatenate([
        rng.integers(1, 15, size=int(n_samples * 0.4)),
        rng.integers(15, 55, size=int(n_samples * 0.3)),
        rng.integers(55, 73, size=n_samples - int(n_samples * 0.4) - int(n_samples * 0.3))
    ])
    rng.shuffle(tenure_raw)
    tenure = tenure_raw

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == "No", "No phone service", rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_samples, p=[0.34, 0.44, 0.22])

    def assign_internet_addon(rate_yes: float):
        return np.where(
            internet_service == "No",
            "No internet service",
            rng.choice(["Yes", "No"], size=n_samples, p=[rate_yes, 1.0 - rate_yes])
        )

    online_security = assign_internet_addon(0.35)
    online_backup = assign_internet_addon(0.40)
    device_protection = assign_internet_addon(0.38)
    tech_support = assign_internet_addon(0.34)
    streaming_tv = assign_internet_addon(0.45)
    streaming_movies = assign_internet_addon(0.46)

    # Contract selection is correlated with tenure
    contract_probs = np.where(tenure < 12, 0.85, np.where(tenure < 36, 0.45, 0.15))
    contract = np.where(
        rng.random(n_samples) < contract_probs,
        "Month-to-month",
        rng.choice(["One year", "Two year"], size=n_samples, p=[0.45, 0.55])
    )

    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # Base monthly charges calculation based on services
    base_charge = 20.0
    base_charge += np.where(phone_service == "Yes", 10.0, 0.0)
    base_charge += np.where(multiple_lines == "Yes", 10.0, 0.0)
    base_charge += np.where(internet_service == "DSL", 25.0, np.where(internet_service == "Fiber optic", 50.0, 0.0))
    base_charge += np.where(online_security == "Yes", 5.0, 0.0)
    base_charge += np.where(online_backup == "Yes", 5.0, 0.0)
    base_charge += np.where(device_protection == "Yes", 5.0, 0.0)
    base_charge += np.where(tech_support == "Yes", 5.0, 0.0)
    base_charge += np.where(streaming_tv == "Yes", 10.0, 0.0)
    base_charge += np.where(streaming_movies == "Yes", 10.0, 0.0)
    noise = rng.normal(0, 2.5, size=n_samples)
    monthly_charges = np.clip(np.round(base_charge + noise, 2), 18.25, 118.75)

    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.96, 1.04, size=n_samples), 2)

    # Realistic churn probability formulation
    # High risk: month-to-month, electronic check, fiber optic without tech support, low tenure
    churn_score = (
        -1.5
        - 0.04 * tenure
        + 0.015 * monthly_charges
        + 1.2 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.6 * (payment_method == "Electronic check")
        + 0.5 * (internet_service == "Fiber optic")
        - 0.4 * (tech_support == "Yes")
        - 0.4 * (online_security == "Yes")
        + 0.25 * (senior_citizen == 1)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-churn_score))
    churn = np.where(rng.random(n_samples) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })
    return df


def load_telco_data(force_download: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads Telco Customer Churn data.
    First checks local cache, then attempts download, and falls back to synthetic generation.
    Returns (DataFrame, metadata_dict).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "IBM Telco Customer Churn (Kaggle/IBM Watson)",
        "is_synthetic": False,
        "target_column": "Churn",
        "description": "Historical telecommunications customer churn dataset."
    }

    if LOCAL_DATA_FILE.exists() and not force_download:
        logger.info(f"Loading local dataset from {LOCAL_DATA_FILE}")
        df = pd.read_csv(LOCAL_DATA_FILE)
        metadata["records_count"] = len(df)
        return df, metadata

    # Attempt download
    success = download_ibm_telco_data(LOCAL_DATA_FILE)
    if success and LOCAL_DATA_FILE.exists():
        df = pd.read_csv(LOCAL_DATA_FILE)
        metadata["records_count"] = len(df)
        return df, metadata

    # Fallback to synthetic
    logger.info("Using synthetic demonstration dataset generator...")
    df = generate_realistic_synthetic_telco_data()
    df.to_csv(LOCAL_DATA_FILE, index=False)
    metadata["is_synthetic"] = True
    metadata["source"] = "Realistic Synthetic Demonstration Data (Engineered matching IBM Telco Distribution)"
    metadata["records_count"] = len(df)
    return df, metadata


def generate_sample_upload_file(df: pd.DataFrame, target_path: Path = SAMPLE_UPLOAD_FILE, n_samples: int = 100) -> Path:
    """Creates a sample test CSV for user batch uploads (excluding the ground truth Churn column)."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = df.drop(columns=["Churn"], errors="ignore").head(n_samples).copy()
    sample_df.to_csv(target_path, index=False)
    logger.info(f"Saved sample upload file with {len(sample_df)} rows to {target_path}")
    return target_path


if __name__ == "__main__":
    df, meta = load_telco_data()
    print("Loaded data shape:", df.shape)
    print("Metadata:", meta)
    sample_path = generate_sample_upload_file(df)
    print("Sample test file generated:", sample_path)
