"""
FinShield Data Generator
Generates realistic, production-like credit risk and loan default application data
with authentic financial correlation structures for regulatory explainability testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_credit_dataset(n_samples: int = 10000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # 1. Base Applicant Demographics & Bureau Data
    applicant_ids = [f"APP-{10000 + i}" for i in range(n_samples)]
    
    # Age: gamma distribution clipped to 21 - 72
    age = np.clip(np.random.gamma(shape=9, scale=4, size=n_samples).astype(int), 21, 72)
    
    # Annual Income: Log-normal distribution ($24,000 to $220,000)
    income_log = np.random.normal(loc=11.0, scale=0.55, size=n_samples)
    annual_income = np.clip(np.exp(income_log), 24000, 250000).round(-2)
    
    # Employment length (correlated with age)
    max_emp = np.maximum(0, age - 20)
    employment_years = np.clip(np.random.beta(a=2, b=4, size=n_samples) * max_emp, 0, 35).round(1)
    
    # Home ownership
    home_ownership_choices = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    home_ownership_probs = [0.46, 0.44, 0.09, 0.01]
    home_ownership = np.random.choice(home_ownership_choices, size=n_samples, p=home_ownership_probs)
    
    # Loan Intent
    loan_intent_choices = [
        'DEBTCONSOLIDATION', 'CREDITCARD_REFINANCE', 'HOMEIMPROVEMENT',
        'SMALL_BUSINESS', 'MEDICAL', 'EDUCATION', 'PERSONAL'
    ]
    loan_intent_probs = [0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03]
    loan_intent = np.random.choice(loan_intent_choices, size=n_samples, p=loan_intent_probs)
    
    # Credit Score (Bureau / FICO equivalent: 320 to 850)
    credit_score = np.clip(
        np.random.normal(loc=680, scale=65, size=n_samples),
        320, 850
    ).astype(int)
    
    # Loan Grade derived from Credit Score with noise
    def assign_grade(score):
        if score >= 750: return 'A'
        elif score >= 700: return 'B'
        elif score >= 650: return 'C'
        elif score >= 600: return 'D'
        elif score >= 550: return 'E'
        else: return 'F'
    loan_grade = [assign_grade(s) for s in credit_score]
    
    # Interest Rate correlated with Grade
    grade_rate_map = {
        'A': (5.5, 7.9),
        'B': (8.0, 11.5),
        'C': (11.6, 14.9),
        'D': (15.0, 18.9),
        'E': (19.0, 23.5),
        'F': (24.0, 29.9)
    }
    interest_rate = np.array([
        np.round(np.random.uniform(*grade_rate_map[g]), 2) for g in loan_grade
    ])
    
    # Loan Amount: ($1,500 to $40,000)
    loan_amount = np.clip(
        np.random.normal(loc=15000, scale=8000, size=n_samples),
        1500, 40000
    ).round(-2)
    
    # Revolving Utilization: correlated inversely with credit score
    # Lower credit score -> higher credit card utilization
    base_util = 1.0 - ((credit_score - 320) / (850 - 320))
    revolving_utilization = np.clip(base_util + np.random.normal(0, 0.15, size=n_samples), 0.02, 1.15).round(3)
    
    # Debt-to-Income (DTI) Ratio: Monthly debt / Monthly income
    # Realistic range 0.05 to 0.65
    monthly_income = annual_income / 12.0
    loan_monthly_payment = (loan_amount * (interest_rate / 100 / 12)) / (1 - (1 + interest_rate / 100 / 12)**(-36))
    existing_debt_monthly = monthly_income * np.clip(np.random.beta(2, 5, size=n_samples), 0.05, 0.45)
    debt_to_income = np.clip((existing_debt_monthly + loan_monthly_payment) / monthly_income, 0.05, 0.70).round(3)
    
    # Open Credit Lines (2 to 32)
    total_open_credit_lines = np.clip(np.random.poisson(lam=11, size=n_samples), 1, 35)
    
    # Delinquencies in past 2 years (poisson, higher for lower credit scores)
    delinq_rate = np.where(credit_score < 620, 1.2, np.where(credit_score < 700, 0.35, 0.05))
    delinquencies_2yrs = np.clip(np.random.poisson(lam=delinq_rate), 0, 7)
    
    # 2. Authentic Default Ground Truth Formula (Latent Risk Modeling)
    # Log-odds of default
    z = (
        - 2.80
        + 4.20 * (debt_to_income - 0.25)
        + 3.50 * (revolving_utilization - 0.40)
        - 0.012 * (credit_score - 650)
        + 0.55 * delinquencies_2yrs
        + 0.08 * (interest_rate - 12.0)
        - 0.04 * employment_years
        + np.random.normal(0, 0.35, size=n_samples) # Unobserved shocks
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    loan_status_default = (np.random.uniform(0, 1, size=n_samples) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'applicant_id': applicant_ids,
        'age': age,
        'annual_income': annual_income,
        'employment_years': employment_years,
        'home_ownership': home_ownership,
        'loan_intent': loan_intent,
        'loan_grade': loan_grade,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'credit_score': credit_score,
        'revolving_utilization': revolving_utilization,
        'debt_to_income': debt_to_income,
        'total_open_credit_lines': total_open_credit_lines,
        'delinquencies_2yrs': delinquencies_2yrs,
        'loan_status_default': loan_status_default
    })
    
    return df

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = generate_credit_dataset(n_samples=10000, random_seed=42)
    output_path = output_dir / "credit_applications.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} records at: {output_path}")
    print(f"Default Rate: {df['loan_status_default'].mean():.2%}")
    print("Feature summary:")
    print(df[['annual_income', 'loan_amount', 'credit_score', 'debt_to_income', 'revolving_utilization']].describe().T[['mean', 'min', '50%', 'max']])
