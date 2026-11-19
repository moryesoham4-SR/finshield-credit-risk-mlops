"""
FinShield Model Training & Calibration Engine
Trains an enterprise Gradient Boosted Trees classifier with 3-fold cross-validated probability calibration.
Logs banking metrics: ROC-AUC, PR-AUC, Gini Coefficient, and Brier Score.
Persists complete serialized inference pipeline.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    f1_score
)
from sklearn.model_selection import train_test_split
from feature_engineering import build_preprocessing_pipeline

def train_and_evaluate(
    data_path: str,
    output_dir: str,
    random_state: int = 42
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading engineered dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    target_col = "loan_status_default"
    drop_cols = ["applicant_id", target_col]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # 1. Stratified Splits: Train/Dev (80%), Held-out Test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=random_state, stratify=y
    )
    
    print(f"Dataset Splits: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Train Default Rate: {y_train.mean():.2%}, Test Default Rate: {y_test.mean():.2%}")
    
    # 2. Build and fit preprocessor
    preprocessor = build_preprocessing_pipeline()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = preprocessor.get_feature_names_out().tolist()
    
    # 3. Model Training & 3-fold Probability Calibration (Banking Standard)
    print("Training Gradient Boosted Decision Trees with 3-Fold Probability Calibration...")
    base_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        random_state=random_state
    )
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model,
        method='sigmoid',
        cv=3
    )
    calibrated_model.fit(X_train_trans, y_train)
    
    # Also train standalone base_model on full train set for raw feature importance attribution
    base_model.fit(X_train_trans, y_train)
    
    # 4. Model Evaluation on Held-Out Test Set
    test_probs = calibrated_model.predict_proba(X_test_trans)[:, 1]
    test_preds = (test_probs >= 0.35).astype(int)  # 35% risk threshold for conservative banking
    
    roc_auc = float(roc_auc_score(y_test, test_probs))
    pr_auc = float(average_precision_score(y_test, test_probs))
    brier = float(brier_score_loss(y_test, test_probs))
    gini = float(2.0 * roc_auc - 1.0)
    f1 = float(f1_score(y_test, test_preds))
    
    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "gini_coefficient": round(gini, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "test_records": len(X_test),
        "test_default_rate": round(float(y_test.mean()), 4)
    }
    
    print("\n" + "="*50)
    print("FINANCIAL REGULATORY MODEL EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
    print("="*50)
    
    # 5. Top Predictive Drivers (Global Importance)
    importances = base_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 7 Global Predictive Risk Drivers:")
    for i in range(min(7, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
    
    # 6. Save Pipeline & Metrics Artifacts
    pipeline_artifact = {
        "preprocessor": preprocessor,
        "base_model": base_model,
        "calibrated_model": calibrated_model,
        "feature_names": feature_names
    }
    
    model_file = output_path / "model_pipeline.joblib"
    joblib.dump(pipeline_artifact, model_file)
    print(f"\nSerialized model pipeline saved to: {model_file}")
    
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to: {metrics_file}")
    
    # 7. Generate Sample Test Applications for Endpoint Simulation
    test_samples = df.loc[X_test.index[:5]].copy()
    test_samples_json = output_path.parent / "data" / "test" / "sample_applications.json"
    test_samples_json.parent.mkdir(parents=True, exist_ok=True)
    test_samples.to_json(test_samples_json, orient="records", indent=2)
    print(f"Saved sample inference applications to: {test_samples_json}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    processed_csv = base_dir / "data" / "processed" / "credit_processed.csv"
    artifacts_dir = base_dir / "artifacts"
    train_and_evaluate(str(processed_csv), str(artifacts_dir))
