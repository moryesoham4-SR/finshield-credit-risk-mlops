"""
FinShield Automated Testing Suite
Runs unit and integration tests across data validation contracts, feature engineering, and inference scoring.
"""

import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_validation import CreditDataValidator
from feature_engineering import FinancialRatioEngineer
from score import init, run

class TestFinShieldPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        init()
        cls.sample_good = {
            "applicant_id": "TEST-GOOD-01",
            "age": 42,
            "annual_income": 125000.0,
            "employment_years": 12.0,
            "home_ownership": "MORTGAGE",
            "loan_intent": "HOMEIMPROVEMENT",
            "loan_grade": "A",
            "loan_amount": 10000.0,
            "interest_rate": 6.8,
            "credit_score": 780,
            "revolving_utilization": 0.12,
            "debt_to_income": 0.18,
            "total_open_credit_lines": 14,
            "delinquencies_2yrs": 0
        }
        
        cls.sample_risky = {
            "applicant_id": "TEST-RISKY-02",
            "age": 24,
            "annual_income": 32000.0,
            "employment_years": 1.0,
            "home_ownership": "RENT",
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "E",
            "loan_amount": 25000.0,
            "interest_rate": 21.5,
            "credit_score": 540,
            "revolving_utilization": 0.88,
            "debt_to_income": 0.58,
            "total_open_credit_lines": 6,
            "delinquencies_2yrs": 3
        }

    def test_01_schema_validation(self):
        """Verify data validator rejects missing columns."""
        df_invalid = pd.DataFrame([{"age": 30}])
        ok, errors = CreditDataValidator.validate_schema(df_invalid)
        self.assertFalse(ok)
        self.assertTrue(len(errors) > 0)

    def test_02_business_rule_bounds(self):
        """Verify business rule violations trigger validation failures."""
        df_corrupt = pd.DataFrame([self.sample_good.copy()])
        df_corrupt.loc[0, "credit_score"] = 999  # Above 850 ceiling
        ok, report = CreditDataValidator.validate_business_rules(df_corrupt)
        self.assertFalse(ok)
        self.assertIn("invalid_credit_score", report["violations"])

    def test_03_financial_ratio_engineering(self):
        """Verify financial ratio transformations are populated."""
        engineer = FinancialRatioEngineer()
        df = pd.DataFrame([self.sample_good])
        out = engineer.transform(df)
        self.assertIn("revolving_stress", out.columns)
        self.assertIn("delinquency_severity", out.columns)
        self.assertIn("payment_burden", out.columns)
        self.assertGreater(out.loc[0, "income_to_loan_ratio"], 1.0)

    def test_04_endpoint_scoring_good_applicant(self):
        """Verify prime applicant is APPROVED with zero adverse actions."""
        payload = json.dumps({"data": self.sample_good})
        raw_resp = run(payload)
        resp = json.loads(raw_resp)
        self.assertEqual(resp["status"], "SUCCESS")
        result = resp["results"][0]
        self.assertEqual(result["underwriting_decision"], "APPROVED")
        self.assertLess(result["risk_probability"], 0.35)
        self.assertEqual(len(result["adverse_action_reasons"]), 0)

    def test_05_endpoint_scoring_risky_applicant(self):
        """Verify subprime applicant is DECLINED with top adverse action notices."""
        payload = json.dumps({"data": self.sample_risky})
        raw_resp = run(payload)
        resp = json.loads(raw_resp)
        self.assertEqual(resp["status"], "SUCCESS")
        result = resp["results"][0]
        self.assertEqual(result["underwriting_decision"], "DECLINED")
        self.assertGreaterEqual(result["risk_probability"], 0.35)
        self.assertGreaterEqual(len(result["adverse_action_reasons"]), 1)

if __name__ == "__main__":
    unittest.main()
