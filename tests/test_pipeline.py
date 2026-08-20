"""
Unit Tests for Credit Card Risk Modeling Pipeline.
Tests data loader, preprocessing, feature engineering, and inference.
"""

import unittest
import numpy as np
import pandas as pd
import os

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer, DEFAULT_SELECTED_FEATURES
from src.data_loader import create_sample_customers
from src.threshold_optimizer import ThresholdOptimizer


class TestCreditRiskPipeline(unittest.TestCase):

    def setUp(self):
        # Create a synthetic mini dataset mimicking UCI schema
        np.random.seed(42)
        n = 50
        self.raw_data = pd.DataFrame({
            "Customer_ID": range(1, n + 1),
            "LIMIT_BAL": np.random.choice([10000, 50000, 100000, 200000], size=n),
            "sex": np.random.choice([1, 2], size=n),
            "education": np.random.choice([0, 1, 2, 3, 4, 5], size=n),
            "marriage": np.random.choice([0, 1, 2, 3], size=n),
            "age": np.random.choice([22, 28, 35, 45, np.nan], size=n),
            "pay_0": np.random.choice([-1, 0, 1, 2], size=n),
            "pay_2": np.random.choice([-1, 0, 1, 2], size=n),
            "pay_3": np.random.choice([-1, 0, 1, 2], size=n),
            "pay_4": np.random.choice([-1, 0, 1, 2], size=n),
            "pay_5": np.random.choice([-1, 0, 1, 2], size=n),
            "pay_6": np.random.choice([-1, 0, 1, 2], size=n),
            "Bill_amt1": np.random.uniform(0, 100000, size=n),
            "Bill_amt2": np.random.uniform(0, 100000, size=n),
            "Bill_amt3": np.random.uniform(0, 100000, size=n),
            "Bill_amt4": np.random.uniform(0, 100000, size=n),
            "Bill_amt5": np.random.uniform(0, 100000, size=n),
            "Bill_amt6": np.random.uniform(0, 100000, size=n),
            "pay_amt1": np.random.uniform(0, 10000, size=n),
            "pay_amt2": np.random.uniform(0, 10000, size=n),
            "pay_amt3": np.random.uniform(0, 10000, size=n),
            "pay_amt4": np.random.uniform(0, 10000, size=n),
            "pay_amt5": np.random.uniform(0, 10000, size=n),
            "pay_amt6": np.random.uniform(0, 10000, size=n),
            "AVG_Bill_amt": np.random.uniform(0, 100000, size=n),
            "PAY_TO_BILL_ratio": np.random.uniform(0, 1, size=n),
            "next_month_default": np.random.choice([0, 1], p=[0.75, 0.25], size=n)
        })

    def test_preprocessing(self):
        preprocessor = DataPreprocessor()
        clean_df = preprocessor.fit_transform(self.raw_data)
        
        # Verify age has no nulls
        self.assertEqual(clean_df["age"].isnull().sum(), 0)
        # Verify marriage is consolidated
        self.assertTrue(set(clean_df["marriage"].unique()).issubset({1, 2, 3}))
        # Verify education is consolidated
        self.assertTrue(set(clean_df["education"].unique()).issubset({1, 2, 3, 4}))

    def test_feature_engineering(self):
        preprocessor = DataPreprocessor()
        clean_df = preprocessor.fit_transform(self.raw_data)
        
        engineer = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
        featured_df = engineer.transform(clean_df)
        
        # Verify all selected features exist
        for col in DEFAULT_SELECTED_FEATURES:
            self.assertIn(col, featured_df.columns)
            self.assertEqual(featured_df[col].isnull().sum(), 0)

        # Verify delinquency score computation
        self.assertTrue((featured_df["delinquent_months"] >= 0).all())
        self.assertTrue((featured_df["credit_utilization"] >= 0).all())

    def test_threshold_optimizer(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.35, 0.65])
        
        optimizer = ThresholdOptimizer(cost_fn=5000.0, cost_fp=500.0)
        results = optimizer.find_optimal_thresholds(y_true, y_prob)
        
        self.assertIn("baseline_threshold_0_5", results)
        self.assertIn("best_f1_threshold", results)
        self.assertIn("best_f2_threshold_risk_averse", results)
        self.assertGreaterEqual(results["recall_pct_improvement"], 0.0)


if __name__ == "__main__":
    unittest.main()
