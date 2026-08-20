"""
End-to-End Training CLI Script for Credit Card Risk Modeling.
Executes:
1. Data loading & validation
2. Missing value imputation & outlier winsorization
3. Domain feature engineering & feature selection
4. Class imbalance mitigation with SMOTE
5. Multi-model benchmarking across 7 algorithms
6. Hyperparameter tuning using GridSearchCV
7. Classification threshold optimization (F1, F2, Cost Matrix)
8. Artifact serialization into models/
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure clean UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_loader import load_raw_data, split_data
from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer, DEFAULT_SELECTED_FEATURES
from src.models import CreditRiskPipeline
from src.threshold_optimizer import ThresholdOptimizer


def run_training_pipeline(tune_model: str = "Random Forest", output_dir: str = "models"):
    print("=" * 70)
    print(" [CREDIT CARD RISK MODELING] END-TO-END TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[STEP 1/7] Loading Raw Dataset...")
    raw_df = load_raw_data()
    print(f"Loaded dataset with shape: {raw_df.shape}")

    # 2. Preprocess & Clean Data
    print("\n[STEP 2/7] Cleaning Data & Outlier Winsorization...")
    preprocessor = DataPreprocessor()
    clean_df = preprocessor.fit_transform(raw_df)

    # 3. Domain Feature Engineering
    print("\n[STEP 3/7] Generating Financial & Delinquency Features...")
    engineer = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
    featured_df = engineer.transform(clean_df)

    # 4. Train/Test Split
    print("\n[STEP 4/7] Performing Stratified Train/Test Split (80/20)...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(featured_df, test_size=0.2, random_state=42)
    
    X_train = X_train_raw[DEFAULT_SELECTED_FEATURES]
    X_test = X_test_raw[DEFAULT_SELECTED_FEATURES]
    print(f"Training features shape: {X_train.shape}, Test features shape: {X_test.shape}")

    # Save processed feature matrices
    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    X_train.assign(target=y_train.values).to_csv(os.path.join(processed_dir, "train_features.csv"), index=False)
    X_test.assign(target=y_test.values).to_csv(os.path.join(processed_dir, "test_features.csv"), index=False)
    print(f"[INFO] Saved processed datasets to: {processed_dir}")

    # 5. Multi-Model Benchmark
    print("\n[STEP 5/7] Benchmarking 7 Machine Learning Algorithms with SMOTE...")
    pipeline = CreditRiskPipeline(random_state=42)
    benchmark_df = pipeline.benchmark_models(X_train, y_train, X_test, y_test)
    
    print("\n" + "-" * 70)
    print(" BASELINE BENCHMARK COMPARISON TABLE (TEST SET)")
    print("-" * 70)
    print(benchmark_df.to_string(index=False))
    print("-" * 70)

    # 6. Hyperparameter Tuning
    print(f"\n[STEP 6/7] Optimizing {tune_model} using GridSearchCV (5-Fold Stratified CV)...")
    best_model, metrics = pipeline.tune_and_fit_best_model(
        X_train, y_train, X_test, y_test, model_type=tune_model
    )

    print("\n" + "=" * 70)
    print(f" OPTIMIZED {tune_model.upper()} TEST METRICS:")
    print("=" * 70)
    for k, v in metrics.items():
        if k not in ["Confusion_Matrix", "Best_Params"]:
            print(f"  * {k:15s}: {v:.4f} ({v*100:.2f}%)")
    print(f"  * Best Parameters: {metrics.get('Best_Params')}")
    print(f"  * Confusion Matrix [TN, FP / FN, TP]: {metrics.get('Confusion_Matrix')}")

    # 7. Threshold Optimization
    print("\n[STEP 7/7] Optimizing Decision Thresholds for Credit Risk Management...")
    y_test_probs = pipeline.predict_proba(X_test)
    optimizer = ThresholdOptimizer(cost_fn=5000.0, cost_fp=500.0)
    opt_summary = optimizer.find_optimal_thresholds(y_test.values, y_test_probs)

    base_t = opt_summary["baseline_threshold_0_5"]
    f2_t = opt_summary["best_f2_threshold_risk_averse"]
    cost_t = opt_summary["min_cost_threshold"]

    print("\n" + "-" * 70)
    print(" [THRESHOLD OPTIMIZATION RESULTS]")
    print("-" * 70)
    print(f"  Baseline (t=0.50)  -> Recall: {base_t['recall']:.4f}, Precision: {base_t['precision']:.4f}, F1: {base_t['f1_score']:.4f}")
    print(f"  Optimal Risk (t={f2_t['threshold']:.2f}) -> Recall: {f2_t['recall']:.4f}, Precision: {f2_t['precision']:.4f}, F2: {f2_t['f2_score']:.4f}")
    print(f"  Min Financial Cost (t={cost_t['threshold']:.2f}) -> Cost: ${cost_t['financial_cost']:,.2f}")
    print(f"  [METRIC] Defaulter Recall Improvement: +{opt_summary['recall_pct_improvement']:.2f}% over standard 0.50 threshold")
    print("-" * 70)

    # Update pipeline threshold
    pipeline.threshold = f2_t["threshold"]
    pipeline.metrics["threshold_optimization"] = {
        "baseline": base_t,
        "optimal_f2": f2_t,
        "min_cost": cost_t,
        "recall_pct_improvement": opt_summary["recall_pct_improvement"]
    }

    # Save Pipeline Artifacts
    pipeline.save(output_dir=output_dir)
    print("\n[COMPLETE] Training pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Credit Card Default Prediction Models")
    parser.add_argument("--model", type=str, default="Random Forest", choices=["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression"])
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save trained model artifacts")
    args = parser.parse_args()

    run_training_pipeline(tune_model=args.model, output_dir=args.output_dir)
