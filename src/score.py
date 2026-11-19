"""
FinShield Azure ML Managed Online Endpoint Inference Script
Production-ready score.py conforming to the Azure ML v2 scoring specification (init + run).
Performs sub-millisecond real-time risk scoring, decision classification, and Adverse Action generation.
"""

import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from feature_engineering import FinancialRatioEngineer
from explainability import AdverseActionExplainer

# Global container state
MODEL_PIPELINE = None
EXPLAINER = None
RATIO_ENGINEER = None

def init():
    """Initializes the model and preprocessor inside the Azure ML container."""
    global MODEL_PIPELINE, EXPLAINER, RATIO_ENGINEER
    
    # Azure ML passes the model path via AZUREML_MODEL_DIR environment variable
    model_dir = os.getenv("AZUREML_MODEL_DIR")
    if model_dir:
        model_path = Path(model_dir) / "model_pipeline.joblib"
    else:
        # Fallback to local artifacts directory for testing
        model_path = Path(__file__).resolve().parent.parent / "artifacts" / "model_pipeline.joblib"
        
    print(f"[AzureML Init] Loading model artifacts from: {model_path}")
    MODEL_PIPELINE = joblib.load(model_path)
    
    base_model = MODEL_PIPELINE["base_model"]
    feature_names = MODEL_PIPELINE["feature_names"]
    
    EXPLAINER = AdverseActionExplainer(
        feature_names=feature_names,
        feature_importances=base_model.feature_importances_
    )
    RATIO_ENGINEER = FinancialRatioEngineer()
    print("[AzureML Init] Model and Explainability engine successfully loaded.")

def run(raw_data: str) -> str:
    """Processes real-time inference requests."""
    global MODEL_PIPELINE, EXPLAINER, RATIO_ENGINEER
    
    try:
        payload = json.loads(raw_data)
        data = payload.get("data", payload)
        
        if isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            return json.dumps({"error": "Invalid payload format. Expected JSON object or list."})
            
        df_raw = pd.DataFrame(records)
        
        # 1. Apply Feature Engineering
        df_engineered = RATIO_ENGINEER.transform(df_raw)
        
        # 2. Transform Features via Preprocessor
        preprocessor = MODEL_PIPELINE["preprocessor"]
        calibrated_model = MODEL_PIPELINE["calibrated_model"]
        
        X_trans = preprocessor.transform(df_engineered)
        
        # 3. Predict Calibrated Probabilities
        probs = calibrated_model.predict_proba(X_trans)[:, 1]
        
        # 4. Generate Explanations & Decisions
        decisions = []
        for i, raw_rec in enumerate(records):
            p = float(probs[i])
            decision = EXPLAINER.explain_decision(raw_rec, risk_prob=p, threshold=0.35)
            decisions.append(decision)
            
        return json.dumps({"status": "SUCCESS", "results": decisions}, indent=2)
        
    except Exception as e:
        return json.dumps({"status": "ERROR", "message": str(e)})

if __name__ == "__main__":
    # Local verification simulating Azure ML Managed Online Endpoint
    print("Testing Azure ML Endpoint Scoring Locally...")
    init()
    
    sample_file = Path(__file__).resolve().parent.parent / "data" / "test" / "sample_applications.json"
    with open(sample_file, "r") as f:
        sample_data = json.load(f)
        
    # Test single application
    print("\n--- SINGLE APPLICATION INFERENCE ---")
    test_req = json.dumps({"data": sample_data[0]})
    resp = run(test_req)
    print(resp)
    
    # Test batch application
    print("\n--- BATCH OF 3 APPLICATIONS INFERENCE ---")
    batch_req = json.dumps({"data": sample_data[:3]})
    resp_batch = run(batch_req)
    print(resp_batch)
