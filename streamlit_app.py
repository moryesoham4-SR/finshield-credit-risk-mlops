"""
FinShield Interactive Web Application
Built with Streamlit for live demonstration to evaluators, professors, and recruiters.
Features interactive loan underwriting, calibrated probability gauge, and ECOA Adverse Action notices.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from feature_engineering import FinancialRatioEngineer
from explainability import AdverseActionExplainer

# Page Configuration
st.set_page_config(
    page_title="FinShield | Enterprise Credit Risk Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
    }
    .adverse-box {
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_artifacts():
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "artifacts" / "model_pipeline.joblib"
    metrics_path = base_dir / "artifacts" / "metrics.json"
    
    pipeline = joblib.load(model_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
        
    engineer = FinancialRatioEngineer()
    explainer = AdverseActionExplainer(
        feature_names=pipeline["feature_names"],
        feature_importances=pipeline["base_model"].feature_importances_
    )
    
    return pipeline, metrics, engineer, explainer

pipeline, metrics, engineer, explainer = load_model_artifacts()

# Title Header
st.markdown('<div class="main-header">🛡️ FinShield: Credit Risk & Explainable AI Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Regulatory-Compliant Credit Underwriting with Calibrated ML & Automated Adverse Action Generation</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📋 Live Credit Underwriter", "📊 Model Metrics & Benchmark", "☁️ Azure Cloud Architecture"])

with tab1:
    st.sidebar.header("🎯 Demo Presets")
    preset = st.sidebar.selectbox(
        "Choose an Applicant Profile:",
        ["Custom Applicant", "Profile A: Prime Borrower (Approval)", "Profile B: Subprime Borrower (Declined + Adverse Action)", "Profile C: Borderline Near-Prime"]
    )
    
    # Preset default values
    if preset == "Profile A: Prime Borrower (Approval)":
        d_age, d_income, d_emp = 42, 125000.0, 12.0
        d_home, d_intent, d_grade = "MORTGAGE", "HOMEIMPROVEMENT", "A"
        d_amount, d_rate = 15000.0, 6.5
        d_score, d_util, d_dti, d_lines, d_delinq = 780, 0.12, 0.18, 14, 0
    elif preset == "Profile B: Subprime Borrower (Declined + Adverse Action)":
        d_age, d_income, d_emp = 24, 32000.0, 1.0
        d_home, d_intent, d_grade = "RENT", "DEBTCONSOLIDATION", "E"
        d_amount, d_rate = 22000.0, 21.5
        d_score, d_util, d_dti, d_lines, d_delinq = 540, 0.88, 0.56, 6, 2
    elif preset == "Profile C: Borderline Near-Prime":
        d_age, d_income, d_emp = 31, 58000.0, 4.0
        d_home, d_intent, d_grade = "RENT", "PERSONAL", "C"
        d_amount, d_rate = 14000.0, 12.5
        d_score, d_util, d_dti, d_lines, d_delinq = 665, 0.42, 0.36, 9, 0
    else:
        d_age, d_income, d_emp = 35, 65000.0, 5.0
        d_home, d_intent, d_grade = "MORTGAGE", "DEBTCONSOLIDATION", "B"
        d_amount, d_rate = 15000.0, 9.5
        d_score, d_util, d_dti, d_lines, d_delinq = 690, 0.35, 0.32, 10, 0

    st.sidebar.divider()
    st.sidebar.subheader("👤 Applicant Demographics")
    age = st.sidebar.number_input("Age", 18, 80, int(d_age))
    annual_income = st.sidebar.number_input("Annual Income ($)", 10000.0, 500000.0, float(d_income), step=5000.0)
    employment_years = st.sidebar.number_input("Employment History (Years)", 0.0, 45.0, float(d_emp), step=0.5)
    home_ownership = st.sidebar.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"], index=["RENT", "MORTGAGE", "OWN", "OTHER"].index(d_home))

    st.sidebar.subheader("💳 Loan Structure")
    loan_intent = st.sidebar.selectbox("Loan Purpose", [
        "DEBTCONSOLIDATION", "CREDITCARD_REFINANCE", "HOMEIMPROVEMENT", "SMALL_BUSINESS", "MEDICAL", "EDUCATION", "PERSONAL"
    ], index=["DEBTCONSOLIDATION", "CREDITCARD_REFINANCE", "HOMEIMPROVEMENT", "SMALL_BUSINESS", "MEDICAL", "EDUCATION", "PERSONAL"].index(d_intent))
    loan_grade = st.sidebar.selectbox("Assessed Loan Grade", ["A", "B", "C", "D", "E", "F"], index=["A", "B", "C", "D", "E", "F"].index(d_grade))
    loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", 1000.0, 50000.0, float(d_amount), step=1000.0)
    interest_rate = st.sidebar.slider("Interest Rate APR (%)", 4.0, 32.0, float(d_rate), step=0.25)

    st.sidebar.subheader("📑 Credit Bureau Profile")
    credit_score = st.sidebar.slider("Credit Bureau Score (FICO)", 300, 850, int(d_score))
    revolving_utilization = st.sidebar.slider("Revolving Utilization Rate", 0.0, 1.20, float(d_util), step=0.01)
    debt_to_income = st.sidebar.slider("Debt-to-Income Ratio (DTI)", 0.05, 0.85, float(d_dti), step=0.01)
    total_open_credit_lines = st.sidebar.slider("Total Open Credit Lines", 1, 35, int(d_lines))
    delinquencies_2yrs = st.sidebar.slider("Delinquencies in Past 2 Years", 0, 8, int(d_delinq))

    # Construct input dataframe
    raw_input = {
        "applicant_id": "APP-LIVE-DEMO",
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "revolving_utilization": revolving_utilization,
        "debt_to_income": debt_to_income,
        "total_open_credit_lines": total_open_credit_lines,
        "delinquencies_2yrs": delinquencies_2yrs
    }
    
    df_raw = pd.DataFrame([raw_input])
    
    # Feature Engineering & Model Inference
    df_engineered = engineer.transform(df_raw)
    preprocessor = pipeline["preprocessor"]
    calibrated_model = pipeline["calibrated_model"]
    
    X_trans = preprocessor.transform(df_engineered)
    risk_prob = float(calibrated_model.predict_proba(X_trans)[0, 1])
    decision_info = explainer.explain_decision(raw_input, risk_prob=risk_prob, threshold=0.35)
    
    # Visual Output Layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Decision & Risk Analysis")
        
        # Decision Banner
        if decision_info["underwriting_decision"] == "APPROVED":
            st.success(f"### ✅ DECISION: APPROVED (Prime Risk Tier)")
            st.write(f"The application successfully meets all risk and debt coverage thresholds.")
        else:
            st.error(f"### ❌ DECISION: DECLINED (Subprime Risk Tier)")
            st.write(f"The calibrated default probability exceeds the risk tolerance ceiling of **35.0%**.")
            
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_prob * 100,
            number = {'suffix': "%", 'valueformat': ".2f"},
            title = {'text': "Calibrated Default Probability", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1F2937"},
                'steps': [
                    {'range': [0, 20], 'color': "#10B981"},    # Low Risk (Green)
                    {'range': [20, 35], 'color': "#FBBF24"},   # Moderate (Yellow)
                    {'range': [35, 100], 'color': "#EF4444"}   # High Risk (Red)
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 35
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("⚖️ Regulatory Adverse Action Compliance")
        
        if decision_info["underwriting_decision"] == "DECLINED":
            st.markdown("""
            <div class="adverse-box">
                <h4 style="color: #991B1B; margin-top:0;">📜 Legally Mandated Adverse Action Notice (ECOA / CFPB)</h4>
                <p style="font-size: 0.9rem; color: #4B5563;">
                    Under the Equal Credit Opportunity Act, applicants must be provided with the primary adverse factors contributing to loan declination:
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            for i, reason in enumerate(decision_info["adverse_action_reasons"], 1):
                st.markdown(f"**Reason #{i}:** {reason['regulatory_notice']}")
        else:
            st.info("ℹ️ **No Adverse Action Required:** The applicant's credit metrics satisfy credit policy standards. Zero derogatory notices filed.")
            
        st.divider()
        st.subheader("💡 Key Financial Engineered Signals")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Payment Burden", f"{df_engineered.loc[0, 'payment_burden']:.1%}")
        col_m2.metric("Revolving Stress", f"{df_engineered.loc[0, 'revolving_stress']:.2f}")
        col_m3.metric("Income-to-Loan", f"{df_engineered.loc[0, 'income_to_loan_ratio']:.1f}x")

with tab2:
    st.header("Model Performance & Regulatory Auditing")
    st.write("Evaluated on a strictly held-out 20% test set (2,000 credit applications).")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']}", "Target: > 0.75")
    col2.metric("Gini Coefficient", f"{metrics['gini_coefficient']}", "Target: > 0.60")
    col3.metric("PR-AUC", f"{metrics['pr_auc']}", "High Precision")
    col4.metric("Brier Score Loss", f"{metrics['brier_score']}", "Well-Calibrated")
    
    st.divider()
    st.subheader("Top Predictive Risk Drivers")
    feature_names = pipeline["feature_names"]
    importances = pipeline["base_model"].feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True).tail(8)
    
    fig_bar = go.Figure(go.Bar(
        x=df_imp["Importance"],
        y=df_imp["Feature"],
        orientation='h',
        marker_color="#2563EB"
    ))
    fig_bar.update_layout(title="Global Feature Importances", height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("Microsoft Azure Production Cloud Architecture")
    st.markdown("""
    FinShield is engineered with **Azure-Native Cloud Architecture**:
    * **Storage:** Azure Data Lake Storage (ADLS) Gen2 with hierarchical namespace.
    * **Model Training & Experimentation:** Azure ML v2 SDK Pipelines with automated probability calibration.
    * **Infrastructure as Code (IaC):** Bicep script (`infra/main.bicep`) declaring zero-idle-cost compute (`min_instances = 0`).
    * **CI/CD:** GitHub Actions workflow executing automated data contract testing and Bicep validation on pull requests.
    """)
    st.code("""
# Azure Cloud Deployment Commands
az group create --name rg-finshield-prod --location eastus
az deployment group create --resource-group rg-finshield-prod --template-file infra/main.bicep
az ml job create --file pipelines/pipeline_job.yaml
    """, language="bash")
