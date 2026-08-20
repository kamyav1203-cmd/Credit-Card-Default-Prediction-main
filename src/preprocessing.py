"""
Preprocessing Module for Credit Card Risk Modeling.
Handles missing value imputation, categorical value standardization,
and outlier winsorization (clipping at 1st and 99th percentiles).
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict


NUMERIC_COLS_FOR_CAPPING = [
    "LIMIT_BAL", "age",
    "Bill_amt1", "Bill_amt2", "Bill_amt3",
    "Bill_amt4", "Bill_amt5", "Bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3",
    "pay_amt4", "pay_amt5", "pay_amt6",
    "AVG_Bill_amt", "PAY_TO_BILL_ratio"
]


class DataPreprocessor:
    """
    Production preprocessor implementing financial domain cleaning rules:
    1. Impute missing age with median.
    2. Consolidate unknown/invalid categorical labels for MARRIAGE & EDUCATION.
    3. Winsorize (cap) extreme numeric outliers using learned percentiles.
    """
    
    def __init__(self):
        self.median_age: Optional[float] = None
        self.outlier_caps: Dict[str, Tuple[float, float]] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Learn median values and 1st/99th percentiles from training data."""
        # 1. Learn median age
        if "age" in df.columns:
            self.median_age = float(df["age"].median())

        # 2. Learn 1st and 99th percentiles for numerical columns
        for col in NUMERIC_COLS_FOR_CAPPING:
            if col in df.columns:
                q1 = float(df[col].quantile(0.01))
                q99 = float(df[col].quantile(0.99))
                self.outlier_caps[col] = (q1, q99)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned transformations and standardizations."""
        data = df.copy()

        # 1. Clean Marriage: 1=married, 2=single, 3=others (0 mapped to 3)
        if "marriage" in data.columns:
            data["marriage"] = data["marriage"].apply(lambda x: x if x in [1, 2, 3] else 3)

        # 2. Clean Education: 1=grad school, 2=university, 3=high school, 4=others (0, 5, 6 mapped to 4)
        if "education" in data.columns:
            data["education"] = data["education"].apply(lambda x: x if x in [1, 2, 3] else 4)

        # 3. Missing Value Imputation
        if "age" in data.columns:
            fill_val = self.median_age if self.median_age is not None else 34.0
            data["age"] = data["age"].fillna(fill_val)

        # 4. Outlier Capping (Winsorization)
        for col, (q1, q99) in self.outlier_caps.items():
            if col in data.columns:
                data[col] = data[col].clip(q1, q99)

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit preprocessor and transform dataset."""
        return self.fit(df).transform(df)


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """Helper function to clean dataset and return fitted preprocessor."""
    preprocessor = DataPreprocessor()
    cleaned_df = preprocessor.fit_transform(df)
    return cleaned_df, preprocessor
