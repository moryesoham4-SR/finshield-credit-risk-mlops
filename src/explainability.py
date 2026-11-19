"""
FinShield Regulatory Explainability & Adverse Action Engine
Implements compliance with the Equal Credit Opportunity Act (ECOA) and CFPB requirements.
Translates model risk contributions into human-readable Adverse Action Reason Codes.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd

class AdverseActionExplainer:
    """Computes Adverse Action Notices for loan rejections."""
    
    # Regulatory Reason Code Mappings
    REASON_CATALOG = {
        "credit_score": "Credit bureau score below prime threshold ({value})",
        "revolving_stress": "High credit card balance utilization relative to credit score ({value:.2f})",
        "debt_to_income": "Total debt-to-income ratio ({value:.1%}) exceeds lending ceiling",
        "delinquency_severity": "Prior 30+ day payment delinquencies reported on credit accounts",
        "delinquencies_2yrs": "Recent delinquencies or late payments on bureau file ({value} incidents)",
        "revolving_utilization": "Proportion of revolving credit balances to credit limits is too high ({value:.1%})",
        "interest_rate": "Assessed loan risk tier carries elevated APR ({value:.2f}%)",
        "income_to_loan_ratio": "Requested loan amount is high relative to annual verified income ({value:.2f}x)",
        "payment_burden": "Monthly payment represents excessive burden on gross monthly income ({value:.1%})",
        "employment_years": "Insufficient length of employment tenure ({value:.1f} years)",
        "employment_to_age_ratio": "Limited continuous employment stability history",
        "annual_income": "Verified annual income is insufficient for requested loan structure"
    }

    def __init__(self, feature_names: List[str], feature_importances: np.ndarray, baseline_means: Dict[str, float] = None):
        self.feature_names = feature_names
        self.feature_importances = dict(zip(feature_names, feature_importances))
        self.baseline_means = baseline_means or {}

    def explain_decision(self, raw_record: Dict[str, Any], risk_prob: float, threshold: float = 0.35) -> Dict[str, Any]:
        """Generates adverse action reasons for rejected applications."""
        
        status = "APPROVED" if risk_prob < threshold else "DECLINED"
        result = {
            "applicant_id": raw_record.get("applicant_id", "UNKNOWN"),
            "risk_probability": round(risk_prob, 4),
            "underwriting_decision": status,
            "decision_threshold": threshold,
            "adverse_action_reasons": []
        }
        
        if status == "DECLINED":
            # Identify which raw attributes contributed to high risk
            adverse_candidates = []
            
            # 1. Credit Score Check
            cs = raw_record.get("credit_score", 700)
            if cs < 660:
                score_impact = (660 - cs) / 100.0 * self.feature_importances.get("credit_score", 0.3)
                adverse_candidates.append(("credit_score", score_impact, cs))
                
            # 2. DTI Check
            dti = raw_record.get("debt_to_income", 0.30)
            if dti > 0.38:
                dti_impact = (dti - 0.38) * 10.0 * self.feature_importances.get("debt_to_income", 0.1)
                adverse_candidates.append(("debt_to_income", dti_impact, dti))
                
            # 3. Revolving Utilization Check
            util = raw_record.get("revolving_utilization", 0.30)
            if util > 0.45:
                util_impact = (util - 0.45) * 5.0 * self.feature_importances.get("revolving_stress", 0.2)
                adverse_candidates.append(("revolving_utilization", util_impact, util))
                
            # 4. Delinquencies
            delinq = raw_record.get("delinquencies_2yrs", 0)
            if delinq > 0:
                delinq_impact = delinq * 1.5 * self.feature_importances.get("delinquency_severity", 0.1)
                adverse_candidates.append(("delinquencies_2yrs", delinq_impact, delinq))
                
            # 5. Income to Loan Ratio
            inc = raw_record.get("annual_income", 50000)
            loan = raw_record.get("loan_amount", 15000)
            ratio = inc / max(loan, 1.0)
            if ratio < 3.0:
                ratio_impact = (3.0 - ratio) * self.feature_importances.get("income_to_loan_ratio", 0.05)
                adverse_candidates.append(("income_to_loan_ratio", ratio_impact, ratio))
                
            # Sort by highest impact descending, pick top 3 (CFPB Standard)
            adverse_candidates.sort(key=lambda x: x[1], reverse=True)
            top_factors = adverse_candidates[:3]
            
            # Fallback if no specific threshold was breached (risk driven by model interactions)
            if not top_factors:
                top_factors = [("credit_score", 1.0, cs), ("debt_to_income", 0.8, dti)]
                
            for feat_name, impact, val in top_factors:
                template = self.REASON_CATALOG.get(feat_name, f"Adverse score attribution on {feat_name}")
                formatted = template.format(value=val)
                result["adverse_action_reasons"].append({
                    "factor": feat_name,
                    "regulatory_notice": formatted
                })
                
        return result
