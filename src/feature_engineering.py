"""
FinShield Feature Engineering Component
Transforms raw financial data into high-signal credit risk indicators.
Produces a reproducible transformer pipeline stored as a joblib artifact.
"""

from pathlib import Path
from typing import Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder

class FinancialRatioEngineer(BaseEstimator, TransformerMixin):
    """Computes specialized credit risk domain ratios."""
    
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        # 1. Income to Loan Amount Ratio
        X_out['income_to_loan_ratio'] = (
            X_out['annual_income'] / np.maximum(X_out['loan_amount'], 1.0)
        ).round(3)
        
        # 2. Revolving Credit Stress (High utilization + Low Bureau score interaction)
        credit_deficit = np.maximum(0, 850 - X_out['credit_score']) / 100.0
        X_out['revolving_stress'] = (X_out['revolving_utilization'] * credit_deficit).round(3)
        
        # 3. Delinquency Severity (Late payments amplified by high DTI)
        X_out['delinquency_severity'] = (
            X_out['delinquencies_2yrs'] * (1.0 + X_out['debt_to_income'])
        ).round(3)
        
        # 4. Employment Stability Ratio
        X_out['employment_to_age_ratio'] = (
            X_out['employment_years'] / np.maximum(X_out['age'] - 18, 1.0)
        ).round(3)
        
        # 5. Estimated Monthly Payment to Monthly Income
        monthly_income = X_out['annual_income'] / 12.0
        r = X_out['interest_rate'] / 100.0 / 12.0
        # 36-month fixed installment formula
        installment = (X_out['loan_amount'] * r) / (1.0 - (1.0 + r)**(-36))
        X_out['payment_burden'] = (installment / monthly_income).round(4)
        
        return X_out

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Builds enterprise scikit-learn column transformer."""
    
    numeric_features = [
        'age', 'annual_income', 'employment_years', 'loan_amount',
        'interest_rate', 'credit_score', 'revolving_utilization',
        'debt_to_income', 'total_open_credit_lines', 'delinquencies_2yrs',
        'income_to_loan_ratio', 'revolving_stress', 'delinquency_severity',
        'employment_to_age_ratio', 'payment_burden'
    ]
    
    ordinal_features = ['loan_grade']
    grade_order = [['A', 'B', 'C', 'D', 'E', 'F']]
    
    nominal_features = ['home_ownership', 'loan_intent']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(categories=grade_order), ordinal_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), nominal_features)
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    return preprocessor

def process_and_save(raw_csv: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(raw_csv)
    print(f"Loaded {len(df)} records for feature transformation.")
    
    # 1. Feature Engineering
    engineer = FinancialRatioEngineer()
    df_engineered = engineer.transform(df)
    
    # Save engineered dataset
    processed_csv = output_path / "credit_processed.csv"
    df_engineered.to_csv(processed_csv, index=False)
    print(f"Engineered data saved to: {processed_csv}")

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    raw_data = base / "data" / "raw" / "credit_applications.csv"
    processed_dir = base / "data" / "processed"
    process_and_save(str(raw_data), str(processed_dir))
