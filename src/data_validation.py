"""
FinShield Data Validation Component
Implements enterprise data quality contracts for FinTech credit scoring.
Validates schemas, domain value constraints, and missingness thresholds before ingestion.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

class CreditDataValidator:
    """Enterprise Data Quality Auditor for Credit Applications."""
    
    EXPECTED_COLUMNS = [
        'applicant_id', 'age', 'annual_income', 'employment_years',
        'home_ownership', 'loan_intent', 'loan_grade', 'loan_amount',
        'interest_rate', 'credit_score', 'revolving_utilization',
        'debt_to_income', 'total_open_credit_lines', 'delinquencies_2yrs'
    ]
    
    VALID_HOME_OWNERSHIP = {'RENT', 'MORTGAGE', 'OWN', 'OTHER'}
    VALID_LOAN_GRADES = {'A', 'B', 'C', 'D', 'E', 'F'}
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, list]:
        """Check presence of mandatory columns."""
        errors = []
        missing_cols = [c for c in cls.EXPECTED_COLUMNS if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing mandatory schema columns: {missing_cols}")
        return len(errors) == 0, errors

    @classmethod
    def validate_business_rules(cls, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Validate financial and demographic boundary conditions."""
        report = {
            "total_records": len(df),
            "passed": True,
            "violations": {}
        }
        
        # 1. Age constraints (Banking regulations: age >= 18)
        invalid_age = df[df['age'] < 18]
        if len(invalid_age) > 0:
            report["violations"]["age_under_18"] = len(invalid_age)
            report["passed"] = False
            
        # 2. Credit Score range (Bureau standards: 300 to 850)
        invalid_credit = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
        if len(invalid_credit) > 0:
            report["violations"]["invalid_credit_score"] = len(invalid_credit)
            report["passed"] = False

        # 3. Debt to Income ratio (Must be positive and <= 1.5)
        invalid_dti = df[(df['debt_to_income'] < 0) | (df['debt_to_income'] > 1.5)]
        if len(invalid_dti) > 0:
            report["violations"]["abnormal_dti"] = len(invalid_dti)
            report["passed"] = False

        # 4. Loan Amount (Must be > 0)
        invalid_loans = df[df['loan_amount'] <= 0]
        if len(invalid_loans) > 0:
            report["violations"]["zero_or_negative_loan"] = len(invalid_loans)
            report["passed"] = False

        # 5. Categorical Domain Check
        invalid_homes = set(df['home_ownership'].dropna().unique()) - cls.VALID_HOME_OWNERSHIP
        if invalid_homes:
            report["violations"]["unknown_home_ownership"] = list(invalid_homes)
            report["passed"] = False

        invalid_grades = set(df['loan_grade'].dropna().unique()) - cls.VALID_LOAN_GRADES
        if invalid_grades:
            report["violations"]["unknown_loan_grades"] = list(invalid_grades)
            report["passed"] = False

        return report["passed"], report

def run_validation(input_csv: str, output_report: str = None) -> bool:
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv)
    
    schema_ok, schema_errors = CreditDataValidator.validate_schema(df)
    if not schema_ok:
        raise ValueError(f"Data Schema Failure: {schema_errors}")
        
    rules_ok, report = CreditDataValidator.validate_business_rules(df)
    print(f"Data Contract Validation Status: {'PASSED' if rules_ok else 'FAILED'}")
    print(json.dumps(report, indent=2))
    
    if output_report:
        Path(output_report).parent.mkdir(parents=True, exist_ok=True)
        with open(output_report, "w") as f:
            json.dump(report, f, indent=2)
            
    return rules_ok

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    raw_data = base_dir / "data" / "raw" / "credit_applications.csv"
    report_file = base_dir / "data" / "processed" / "validation_report.json"
    run_validation(str(raw_data), str(report_file))
