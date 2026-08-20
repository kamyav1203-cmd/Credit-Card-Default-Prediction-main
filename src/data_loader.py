"""
Data Loader Module for Credit Card Default Prediction.
Handles dataset loading, raw data standardization, synthetic sample generation for testing,
and train/validation/test splitting.
"""

import os
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "default_of_credit_card_clients.csv")
SAMPLE_DATA_PATH = os.path.join(DATA_DIR, "sample_customers.csv")
UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


COLUMN_MAPPING = {
    "ID": "Customer_ID",
    "LIMIT_BAL": "LIMIT_BAL",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    "PAY_0": "pay_0",
    "PAY_1": "pay_0",
    "PAY_2": "pay_2",
    "PAY_3": "pay_3",
    "PAY_4": "pay_4",
    "PAY_5": "pay_5",
    "PAY_6": "pay_6",
    "BILL_AMT1": "Bill_amt1",
    "BILL_AMT2": "Bill_amt2",
    "BILL_AMT3": "Bill_amt3",
    "BILL_AMT4": "Bill_amt4",
    "BILL_AMT5": "Bill_amt5",
    "BILL_AMT6": "Bill_amt6",
    "PAY_AMT1": "pay_amt1",
    "PAY_AMT2": "pay_amt2",
    "PAY_AMT3": "pay_amt3",
    "PAY_AMT4": "pay_amt4",
    "PAY_AMT5": "pay_amt5",
    "PAY_AMT6": "pay_amt6",
    "default payment next month": "next_month_default",
    "default.payment.next.month": "next_month_default"
}


def download_uci_dataset(dest_path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Downloads and extracts the UCI Credit Card Default dataset if not present locally.
    Standardizes column names and computes baseline aggregates.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Check if raw CSV already exists
    if os.path.exists(dest_path):
        print(f"[INFO] Loading existing raw dataset from: {dest_path}")
        return pd.read_csv(dest_path)

    # Check if temp extracted xls exists
    temp_xls = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_temp", "default of credit card clients.xls")
    if os.path.exists(temp_xls):
        print(f"[INFO] Reading dataset from local cache: {temp_xls}")
        df = pd.read_excel(temp_xls, header=1)
    else:
        print(f"[INFO] Downloading dataset from UCI repository...")
        req = urllib.request.Request(UCI_DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                excel_files = [f for f in z.namelist() if f.endswith(".xls") or f.endswith(".xlsx")]
                if not excel_files:
                    raise FileNotFoundError("No Excel file found in downloaded UCI zip.")
                excel_name = excel_files[0]
                with z.open(excel_name) as f:
                    df = pd.read_excel(f, header=1)

    # Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)

    # Compute baseline bill and payment summaries if not already present
    bill_cols = [f"Bill_amt{i}" for i in range(1, 7)]
    pay_cols = [f"pay_amt{i}" for i in range(1, 7)]
    
    df["AVG_Bill_amt"] = df[bill_cols].mean(axis=1)
    total_bill = df[bill_cols].sum(axis=1)
    total_pay = df[pay_cols].sum(axis=1)
    df["PAY_TO_BILL_ratio"] = total_pay / (total_bill + 1e-6)

    # Save to CSV
    df.to_csv(dest_path, index=False)
    print(f"[SUCCESS] Dataset standardized and saved to: {dest_path} (Shape: {df.shape})")
    
    # Create sample customers CSV for testing and UI demo
    create_sample_customers(df, SAMPLE_DATA_PATH)
    return df


def create_sample_customers(df: pd.DataFrame, sample_path=SAMPLE_DATA_PATH, n_samples=10):
    """Generates a small batch of representative customer records for fast testing & UI demo."""
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    # Select both defaulters and non-defaulters
    defaulters = df[df["next_month_default"] == 1].head(n_samples // 2)
    non_defaulters = df[df["next_month_default"] == 0].head(n_samples // 2)
    sample_df = pd.concat([defaulters, non_defaulters], ignore_index=True)
    
    # Add human-readable profile tags for testing
    sample_df["Profile_Description"] = [
        "High Utilization, 2-Month Delinquent (High Risk)",
        "Frequent Delinquencies, Minimal Payments (High Risk)",
        "Late Payment History, Declining Trend (High Risk)",
        "High Credit Utilization, Small Repayments (Medium Risk)",
        "Sporadic Late Payments (Medium Risk)",
        "On-Time Payer, Stable Graduate (Low Risk)",
        "High Limit, Low Utilization, Full Payer (Low Risk)",
        "Zero Delinquencies, High Repayment Ratio (Low Risk)",
        "Consistent Early Payer, Low Bill Gaps (Low Risk)",
        "Established Professional, Reliable History (Low Risk)"
    ]
    
    sample_df.to_csv(sample_path, index=False)
    print(f"[INFO] Created sample test customer dataset at: {sample_path}")


def load_raw_data(data_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw dataset, downloading from UCI if not found."""
    if not os.path.exists(data_path):
        return download_uci_dataset(data_path)
    return pd.read_csv(data_path)


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Splits dataframe into features X and target y, then train/test splits with stratification."""
    target_col = "next_month_default"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = download_uci_dataset()
    print("Dataset Info:")
    print(df.info())
    print("\nDefault Distribution:")
    print(df["next_month_default"].value_counts(normalize=True))
