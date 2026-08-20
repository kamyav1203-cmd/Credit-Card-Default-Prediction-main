"""
Decision Threshold Optimization Module for Credit Risk Assessment.
Calculates optimal operating cutoffs balancing:
- Recall (catching true defaulters to prevent financial loss)
- Precision (limiting unnecessary friction for safe customers)
- F2-score maximization and custom cost matrix minimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score, confusion_matrix


class ThresholdOptimizer:
    """
    Evaluates probability cutoffs and determines optimal operating points.
    """

    def __init__(self, cost_fn: float = 5000.0, cost_fp: float = 500.0):
        """
        Args:
            cost_fn: Cost of a False Negative (unrecovered default / bad loan).
            cost_fp: Cost of a False Positive (operational manual check or friction).
        """
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp

    def evaluate_threshold_grid(
        self, y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Evaluates metrics across candidate threshold values."""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 91)

        records = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
            
            # Financial cost calculation
            cost = (fn * self.cost_fn) + (fp * self.cost_fp)
            
            records.append({
                "threshold": round(float(t), 3),
                "precision": round(float(p), 4),
                "recall": round(float(r), 4),
                "f1_score": round(float(f1), 4),
                "f2_score": round(float(f2), 4),
                "financial_cost": round(float(cost), 2),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn)
            })

        return pd.DataFrame(records)

    def find_optimal_thresholds(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Identifies optimal cutoff points under different business objectives."""
        df_eval = self.evaluate_threshold_grid(y_true, y_prob)

        # 1. Baseline threshold at 0.50
        baseline_row = df_eval.iloc[(df_eval["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

        # 2. Best F1 threshold
        best_f1_row = df_eval.loc[df_eval["f1_score"].idxmax()]

        # 3. Best F2 threshold (Prioritizes Recall for Risk Reduction)
        best_f2_row = df_eval.loc[df_eval["f2_score"].idxmax()]

        # 4. Minimum Financial Cost threshold
        min_cost_row = df_eval.loc[df_eval["financial_cost"].idxmin()]

        # Recall improvement calculation
        recall_baseline = baseline_row["recall"]
        recall_optimized = best_f2_row["recall"]
        recall_pct_improvement = ((recall_optimized - recall_baseline) / (recall_baseline + 1e-6)) * 100.0

        return {
            "baseline_threshold_0_5": baseline_row.to_dict(),
            "best_f1_threshold": best_f1_row.to_dict(),
            "best_f2_threshold_risk_averse": best_f2_row.to_dict(),
            "min_cost_threshold": min_cost_row.to_dict(),
            "recall_pct_improvement": round(recall_pct_improvement, 2),
            "threshold_curve_data": df_eval
        }
