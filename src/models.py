"""
Model Training, Benchmarking, and Evaluation Module.
Implements:
- Class imbalance mitigation via SMOTE
- Multi-model evaluation across 7 ML algorithms
- GridSearchCV hyperparameter optimization
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1, F2, ROC-AUC)
- Model serialization & artifact management
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, roc_auc_score, confusion_matrix, classification_report
)


MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics including F1, F2 and ROC-AUC."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "F2 Score": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    }
    if y_prob is not None:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["ROC-AUC"] = 0.0
    return metrics


def get_base_models(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates base classifiers for benchmarking."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "AdaBoost": AdaBoostClassifier(random_state=random_state),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }


def get_hyperparameter_grids() -> Dict[str, Dict[str, List[Any]]]:
    """Returns search grids for GridSearchCV optimization."""
    return {
        "Random Forest": {
            "n_estimators": [100, 150],
            "max_depth": [10, 15, None],
            "min_samples_split": [2, 5]
        },
        "Gradient Boosting": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5]
        },
        "XGBoost": {
            "n_estimators": [100, 150],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth": [3, 6]
        },
        "Logistic Regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"]
        }
    }


class CreditRiskPipeline:
    """
    End-to-end Machine Learning training and scoring pipeline.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=random_state)
        self.model: Optional[Any] = None
        self.selected_features: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.threshold: float = 0.5

    def balance_and_scale_train(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Applies SMOTE to balance classes, then fits and applies StandardScaler."""
        print(f"[INFO] Train class counts before SMOTE: {dict(pd.Series(y_train).value_counts())}")
        X_res, y_res = self.smote.fit_resample(X_train, y_train)
        print(f"[INFO] Train class counts after SMOTE: {dict(pd.Series(y_res).value_counts())}")
        
        self.selected_features = list(X_train.columns)
        X_res_scaled = self.scaler.fit_transform(X_res)
        return X_res_scaled, y_res.values

    def transform_test(self, X_test: pd.DataFrame) -> np.ndarray:
        """Scales test data using fitted StandardScaler."""
        return self.scaler.transform(X_test[self.selected_features])

    def benchmark_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        """Trains and compares all 7 base classifiers on test data."""
        X_tr_scaled, y_tr_res = self.balance_and_scale_train(X_train, y_train)
        X_ts_scaled = self.transform_test(X_test)

        models = get_base_models(self.random_state)
        results = []

        for name, model in models.items():
            print(f"[TRAINING] Training base model: {name}...")
            model.fit(X_tr_scaled, y_tr_res)
            
            y_pred = model.predict(X_ts_scaled)
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_ts_scaled)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_ts_scaled)

            m = compute_metrics(y_test.values, y_pred, y_prob)
            m["Model"] = name
            results.append(m)

        df_results = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
        return df_results

    def tune_and_fit_best_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
        model_type: str = "Random Forest"
    ) -> Tuple[Any, Dict[str, float]]:
        """Optimizes chosen model using GridSearchCV with 5-fold Stratified Cross-Validation."""
        X_tr_scaled, y_tr_res = self.balance_and_scale_train(X_train, y_train)
        X_ts_scaled = self.transform_test(X_test)

        grids = get_hyperparameter_grids()
        param_grid = grids.get(model_type, {})

        if model_type == "Random Forest":
            base_estimator = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        elif model_type == "XGBoost":
            base_estimator = XGBClassifier(eval_metric="logloss", random_state=self.random_state, n_jobs=-1)
        elif model_type == "Gradient Boosting":
            base_estimator = GradientBoostingClassifier(random_state=self.random_state)
        else:
            base_estimator = LogisticRegression(max_iter=2000, random_state=self.random_state)

        print(f"[TUNING] Running GridSearchCV for {model_type}...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_tr_scaled, y_tr_res)

        self.model = grid_search.best_estimator_
        print(f"[TUNING] Best Params: {grid_search.best_params_}")

        # Evaluate on test set
        y_pred = self.model.predict(X_ts_scaled)
        y_prob = self.model.predict_proba(X_ts_scaled)[:, 1] if hasattr(self.model, "predict_proba") else None
        
        self.metrics = compute_metrics(y_test.values, y_pred, y_prob)
        self.metrics["Best_Params"] = grid_search.best_params_
        self.metrics["Confusion_Matrix"] = confusion_matrix(y_test.values, y_pred).tolist()

        return self.model, self.metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted probability of default."""
        if self.model is None:
            raise ValueError("Model is not fitted. Train or load a model first.")
        X_scaled = self.scaler.transform(X[self.selected_features])
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary predictions at specified decision threshold."""
        thresh = threshold if threshold is not None else self.threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def save(self, output_dir: str = MODELS_DIR):
        """Serializes model pipeline, scaler, feature list and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "best_rf_model.pkl")
        scaler_path = os.path.join(output_dir, "scaler.pkl")
        features_path = os.path.join(output_dir, "selected_features.json")
        metadata_path = os.path.join(output_dir, "model_metadata.json")

        joblib.dump(self.model, model_path, compress=3)
        joblib.dump(self.scaler, scaler_path)
        
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump(self.selected_features, f, indent=2)

        metadata = {
            "metrics": self.metrics,
            "threshold": self.threshold,
            "num_features": len(self.selected_features),
            "features": self.selected_features
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SUCCESS] Model artifacts successfully saved to: {output_dir}")

    @classmethod
    def load(cls, model_dir: str = MODELS_DIR) -> "CreditRiskPipeline":
        """Loads serialized pipeline artifacts."""
        instance = cls()
        model_path = os.path.join(model_dir, "best_rf_model.pkl")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        features_path = os.path.join(model_dir, "selected_features.json")
        metadata_path = os.path.join(model_dir, "model_metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        instance.model = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        
        with open(features_path, "r", encoding="utf-8") as f:
            instance.selected_features = json.load(f)

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                instance.metrics = metadata.get("metrics", {})
                instance.threshold = metadata.get("threshold", 0.5)

        return instance
