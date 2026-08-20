"""
Feature Engineering Module for Credit Card Risk Modeling.
Computes domain-specific credit risk signals:
- Delinquency velocities and weighted recency scores
- Credit utilization and age-to-limit ratios
- Repayment shortfall gaps and trends
- Payment stability metrics (mean, std, coefficient of variation)
- Bill and payment trajectory slopes using linear regression
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import List, Optional


PAY_COLS = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
BILL_COLS = [f"Bill_amt{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"pay_amt{i}" for i in range(1, 7)]
WEIGHTS = np.array([6, 5, 4, 3, 2, 1])

DEFAULT_SELECTED_FEATURES = [
    "mean_delinquency",
    "weighted_pay_score",
    "marriage_education",
    "max_delinquency",
    "pay_0",
    "credit_utilization",
    "pay_2",
    "Bill_amt1",
    "payment_mean",
    "AVG_Bill_amt",
    "delinquent_months",
    "pay_amt2",
    "pay_amt1",
    "pay_3",
    "age_to_limit",
    "Bill_amt2",
    "payment_cv",
    "pay_amt3",
    "bill_trend",
    "payment_std"
]


def _calc_slope(row_values: np.ndarray) -> float:
    """Computes linear regression slope across 6 time points (1 to 6)."""
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y = np.asarray(row_values, dtype=float)
    if np.all(y == y[0]):
        return 0.0
    slope, _, _, _, _ = linregress(x, y)
    return float(slope)


class FeatureEngineer:
    """
    Stateful feature engineer to compute all credit risk indicators
    and extract selected feature subsets for modeling.
    """

    def __init__(self, selected_features: Optional[List[str]] = None):
        self.selected_features = selected_features or DEFAULT_SELECTED_FEATURES
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit feature engineer (learns configuration)."""
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all engineered financial features."""
        data = df.copy()

        # 1. Base Aggregates (if missing)
        if "AVG_Bill_amt" not in data.columns and all(c in data.columns for c in BILL_COLS):
            data["AVG_Bill_amt"] = data[BILL_COLS].mean(axis=1)

        # 2. Credit Utilization Ratio
        if "AVG_Bill_amt" in data.columns and "LIMIT_BAL" in data.columns:
            data["credit_utilization"] = data["AVG_Bill_amt"] / (data["LIMIT_BAL"] + 1e-6)

        # 3. Delinquency Features
        if all(c in data.columns for c in PAY_COLS):
            pay_matrix = data[PAY_COLS].values
            data["delinquent_months"] = np.sum(pay_matrix >= 1, axis=1)
            data["max_delinquency"] = np.max(pay_matrix, axis=1)
            data["mean_delinquency"] = np.mean(pay_matrix, axis=1)
            # Weighted Recency Score (Month 0 weighted 6x, Month 6 weighted 1x)
            data["weighted_pay_score"] = np.dot(pay_matrix, WEIGHTS)

        # 4. Repayment Shortfall Gaps
        if all(c in data.columns for c in BILL_COLS) and all(c in data.columns for c in PAY_AMT_COLS):
            gap_values = data[BILL_COLS].values - data[PAY_AMT_COLS].values
            data["avg_payment_gap"] = np.mean(gap_values, axis=1)
            data["total_payment_gap"] = np.sum(gap_values, axis=1)

        # 5. Payment Consistency Statistics
        if all(c in data.columns for c in PAY_AMT_COLS):
            pay_amt_matrix = data[PAY_AMT_COLS].values
            data["payment_mean"] = np.mean(pay_amt_matrix, axis=1)
            data["payment_std"] = np.std(pay_amt_matrix, axis=1)
            data["payment_cv"] = data["payment_std"] / (data["payment_mean"] + 1e-6)

        # 6. Bill and Payment Trends (Slope over 6 billing cycles)
        if all(c in data.columns for c in BILL_COLS):
            data["bill_trend"] = data[BILL_COLS].apply(lambda row: _calc_slope(row.values), axis=1)
        if all(c in data.columns for c in PAY_AMT_COLS):
            data["pay_trend"] = data[PAY_AMT_COLS].apply(lambda row: _calc_slope(row.values), axis=1)

        # 7. Demographic & Limit Interactions
        if "age" in data.columns and "LIMIT_BAL" in data.columns:
            data["age_to_limit"] = data["age"] / (data["LIMIT_BAL"] + 1e-6)
        if "marriage" in data.columns and "education" in data.columns:
            # Numeric interaction code (e.g. 1_2 -> 12) for tree models and linear models
            data["marriage_education"] = data["marriage"].astype(int) * 10 + data["education"].astype(int)

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform dataset."""
        return self.fit(df).transform(df)

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts only the selected top modeling features."""
        transformed = self.transform(df)
        missing = [f for f in self.selected_features if f not in transformed.columns]
        if missing:
            raise ValueError(f"Missing required features in transformed data: {missing}")
        return transformed[self.selected_features]


def create_feature_pipeline(df: pd.DataFrame, selected_features: Optional[List[str]] = None) -> pd.DataFrame:
    """Convenience function to run end-to-end feature engineering."""
    fe = FeatureEngineer(selected_features=selected_features)
    return fe.fit_transform(df)
