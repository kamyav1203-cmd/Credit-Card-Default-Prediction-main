import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure clean UTF-8 stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from src.models import CreditRiskPipeline
from src.threshold_optimizer import ThresholdOptimizer


def generate_evaluation_report(
    test_features_csv: str = "data/processed/test_features.csv",
    model_dir: str = "models"
):
    if not os.path.exists(test_features_csv):
        raise FileNotFoundError(f"Processed test features not found at: {test_features_csv}. Please run train.py first.")

    print("=" * 75)
    print(" [CREDIT CARD RISK MODEL] EVALUATION & BENCHMARK REPORT")
    print("=" * 75)

    df_test = pd.read_csv(test_features_csv)
    if "target" not in df_test.columns:
        raise ValueError("Column 'target' not found in test dataset.")

    y_test = df_test["target"].values
    X_test = df_test.drop(columns=["target"])

    pipeline = CreditRiskPipeline.load(model_dir=model_dir)
    print(f"\n[INFO] Loaded Production Model from: {model_dir}")
    print(f"[INFO] Features evaluated ({len(pipeline.selected_features)} total): {pipeline.selected_features}")

    # Probabilities & Predictions
    y_probs = pipeline.predict_proba(X_test)
    y_preds_default = (y_probs >= 0.50).astype(int)
    y_preds_opt = pipeline.predict(X_test)

    print("\n" + "-" * 75)
    print(" 1. CLASSIFICATION REPORT (Baseline Threshold = 0.50)")
    print("-" * 75)
    print(classification_report(y_test, y_preds_default, target_names=["Non-Defaulter (0)", "Defaulter (1)"], digits=4))

    print("\n" + "-" * 75)
    print(f" 2. CLASSIFICATION REPORT (Optimized Threshold = {pipeline.threshold:.2f})")
    print("-" * 75)
    print(classification_report(y_test, y_preds_opt, target_names=["Non-Defaulter (0)", "Defaulter (1)"], digits=4))

    roc_auc = roc_auc_score(y_test, y_probs)
    print(f"[SCORE] ROC-AUC Score: {roc_auc:.4f}")

    # Confusion Matrix
    cm_base = confusion_matrix(y_test, y_preds_default)
    cm_opt = confusion_matrix(y_test, y_preds_opt)

    print("\n" + "-" * 75)
    print(" 3. CONFUSION MATRIX COMPARISON")
    print("-" * 75)
    print(f"  Baseline (t=0.50)  : TN={cm_base[0,0]}, FP={cm_base[0,1]} | FN={cm_base[1,0]}, TP={cm_base[1,1]}")
    print(f"  Optimized (t={pipeline.threshold:.2f}) : TN={cm_opt[0,0]}, FP={cm_opt[0,1]} | FN={cm_opt[1,0]}, TP={cm_opt[1,1]}")
    
    caught_more = cm_opt[1,1] - cm_base[1,1]
    print(f"  [IMPACT] Additional Defaulters Caught: +{caught_more} accounts ({caught_more / len(y_test) * 100:.2f}% improvement in loss prevention)")
    print("-" * 75)

    # Saved Metadata Summary
    metadata_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print("\n" + "-" * 75)
        print("💾 4. SAVED MODEL METADATA")
        print("-" * 75)
        print(json.dumps(meta, indent=2))
        print("-" * 75)


if __name__ == "__main__":
    generate_evaluation_report()
