"""
Inference CLI Script for Credit Card Default Risk Prediction.
Supports:
1. Single customer prediction from command line parameters.
2. Batch scoring from CSV input with formatted output export.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Ensure clean UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.preprocessing import DataPreprocessor
from src.features import FeatureEngineer, DEFAULT_SELECTED_FEATURES
from src.models import CreditRiskPipeline


def predict_batch_csv(input_csv: str, output_csv: Optional[str] = None, threshold: Optional[float] = None, model_dir: str = "models"):
    """Scores a batch CSV file of customer profiles."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    print(f"[INFO] Loading customer records from: {input_csv}")
    df_raw = pd.read_csv(input_csv)

    # 1. Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df_raw)

    # 2. Engineer Features
    engineer = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
    df_features = engineer.transform(df_clean)

    # 3. Load Trained Pipeline
    pipeline = CreditRiskPipeline.load(model_dir=model_dir)
    thresh = threshold if threshold is not None else pipeline.threshold

    # 4. Predict Risk Probabilities & Binary Labels
    probs = pipeline.predict_proba(df_features)
    preds = (probs >= thresh).astype(int)

    # 5. Format Results
    results_df = df_raw.copy()
    results_df["Default_Probability"] = np.round(probs, 4)
    results_df["Risk_Score_Pct"] = np.round(probs * 100, 2)
    results_df["Predicted_Default"] = preds
    results_df["Risk_Category"] = np.where(probs >= 0.65, "High Risk", np.where(probs >= 0.35, "Medium Risk", "Low Risk"))

    print("\n" + "=" * 80)
    print(f" BATCH PREDICTION RESULTS (Operating Threshold: {thresh:.2f})")
    print("=" * 80)
    
    display_cols = ["Customer_ID", "LIMIT_BAL", "age", "pay_0", "Default_Probability", "Risk_Category", "Predicted_Default"]
    existing_cols = [c for c in display_cols if c in results_df.columns]
    print(results_df[existing_cols].head(15).to_string(index=False))

    if output_csv:
        results_df.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Full predictions saved to: {output_csv}")

    return results_df


def predict_single_customer(
    limit_bal: float, age: float, sex: int, education: int, marriage: int,
    pay_0: int, pay_2: int, pay_3: int, pay_4: int, pay_5: int, pay_6: int,
    bill_amt1: float, bill_amt2: float, bill_amt3: float, bill_amt4: float, bill_amt5: float, bill_amt6: float,
    pay_amt1: float, pay_amt2: float, pay_amt3: float, pay_amt4: float, pay_amt5: float, pay_amt6: float,
    threshold: Optional[float] = None, model_dir: str = "models"
):
    """Scores a single customer profile."""
    record = {
        "Customer_ID": 99999,
        "LIMIT_BAL": limit_bal,
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": age,
        "pay_0": pay_0,
        "pay_2": pay_2,
        "pay_3": pay_3,
        "pay_4": pay_4,
        "pay_5": pay_5,
        "pay_6": pay_6,
        "Bill_amt1": bill_amt1,
        "Bill_amt2": bill_amt2,
        "Bill_amt3": bill_amt3,
        "Bill_amt4": bill_amt4,
        "Bill_amt5": bill_amt5,
        "Bill_amt6": bill_amt6,
        "pay_amt1": pay_amt1,
        "pay_amt2": pay_amt2,
        "pay_amt3": pay_amt3,
        "pay_amt4": pay_amt4,
        "pay_amt5": pay_amt5,
        "pay_amt6": pay_amt6,
        "AVG_Bill_amt": np.mean([bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]),
        "PAY_TO_BILL_ratio": np.sum([pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]) / (np.sum([bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]) + 1e-6)
    }

    df_single = pd.DataFrame([record])
    
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.fit_transform(df_single)
    
    engineer = FeatureEngineer(selected_features=DEFAULT_SELECTED_FEATURES)
    df_features = engineer.transform(df_clean)

    pipeline = CreditRiskPipeline.load(model_dir=model_dir)
    thresh = threshold if threshold is not None else pipeline.threshold
    
    prob = float(pipeline.predict_proba(df_features)[0])
    pred = int(prob >= thresh)
    risk_level = "High Risk" if prob >= 0.65 else ("Medium Risk" if prob >= 0.35 else "Low Risk")

    print("\n" + "=" * 60)
    print(" [CREDIT RISK ASSESSMENT RESULT]")
    print("=" * 60)
    print(f"  * Credit Limit       : ${limit_bal:,.2f}")
    print(f"  * Age / Demographics : {age} yrs | Education={education} | Marriage={marriage}")
    print(f"  * Most Recent Status : pay_0 = {pay_0} ({'Late' if pay_0 > 0 else 'On Time'})")
    print(f"  * Default Probability: {prob:.4f} ({prob*100:.2f}%)")
    print(f"  * Operating Threshold: {thresh:.2f}")
    print(f"  * Assessment Decision: {'[HIGH RISK] DEFAULT DETECTED' if pred == 1 else '[LOW RISK] APPROVED'}")
    print(f"  * Risk Category      : {risk_level}")
    print("=" * 60)

    return {"probability": prob, "prediction": pred, "risk_category": risk_level}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Credit Card Default Risk")
    parser.add_argument("--input", type=str, help="Path to input CSV file for batch scoring")
    parser.add_argument("--output", type=str, default="data/predictions.csv", help="Path to save output CSV")
    parser.add_argument("--threshold", type=float, default=None, help="Custom decision threshold override (0.0 to 1.0)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory containing trained model artifacts")
    
    args = parser.parse_args()

    if args.input:
        predict_batch_csv(args.input, args.output, args.threshold, args.model_dir)
    else:
        # Default test run with sample customer data
        sample_path = os.path.join("data", "sample_customers.csv")
        if os.path.exists(sample_path):
            predict_batch_csv(sample_path, args.output, args.threshold, args.model_dir)
        else:
            print("[INFO] No input specified. Run with '--input data/sample_customers.csv' or '--help'.")
