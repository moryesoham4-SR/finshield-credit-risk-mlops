# FinShield: Autonomous Credit Risk Underwriting & Regulatory MLOps Platform

[![Live Terminal](https://img.shields.io/badge/Live%20Terminal-Vercel%20Production-000?logo=vercel)](https://finshield-credit-risk-mlops.vercel.app)
[![Tests](https://img.shields.io/badge/Tests-5%20Passed-brightgreen)](tests/test_pipeline.py)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8716-blue)](artifacts/metrics.json)
[![Gini](https://img.shields.io/badge/Gini-0.7432-blueviolet)](artifacts/metrics.json)
[![Latency](https://img.shields.io/badge/Latency-0.48%20ms%20(Sub--ms)-emerald)](index.html)
[![Cloud](https://img.shields.io/badge/Cloud-Microsoft%20Azure-0089D6?logo=microsoftazure)](infra/main.bicep)
[![Regulatory](https://img.shields.io/badge/Compliance-RBI%20Fair%20Practices%20%26%20ECOA-indigo)](index.html)
[![License](https://img.shields.io/badge/License-Apache--2.0-lightgrey)](LICENSE)

An enterprise-grade, end-to-end Machine Learning Operations (MLOps) platform built for **FinTech, Banking, and Institutional Credit Underwriting**. FinShield automates retail credit decisions using **Calibrated Gradient Boosted Decision Trees (GBDT)** with **3-fold Sigmoid Platt Calibration**, achieving sub-millisecond (**0.48 ms**) inference latency, automated **Population Stability Index (PSI)** data drift detection, **Four-Fifths (80%) Rule** algorithmic fairness validation, and legally compliant **Adverse Action Reason Codes** adhering to **RBI Fair Practices Code** and **US ECOA (Regulation B)** standards.

---

## 🏛️ Executive Summary & Key Highlights

* **High-Performance Modeling:** Audited on 10,000+ credit applications, delivering **ROC-AUC of 0.8716**, **Gini of 0.7432**, and minimizing Brier score loss to **0.0878** for precise Expected Loss ($EL = PD \times LGD \times EAD$) capital adequacy.
* **High-Velocity Streaming Feed:** Live simulated ingestion processing **120 applications/min (0.5s frequency)** with sub-millisecond (**0.48 ms**) scoring response and one-click **CSV Audit Trail Export**.
* **Legally Compliant Explainability:** Automatically extracts the top principal adverse factors for declined borrowers, generating printable/downloadable **Official RBI Credit Sanction Letters** and **Mandated Adverse Action Notices** with cryptographic decision hashes.
* **Continuous MLOps Governance:** Real-time **Population Stability Index (PSI: 0.034)** drift monitoring and **Disparate Impact Ratio (DIR: 0.949)** algorithmic fairness auditing.
* **Zero-Idle-Cost Cloud Footprint:** Infrastructure as Code declared in **Azure Bicep (`infra/main.bicep`)** with `minNodeCount: 0` compute auto-scaling, eliminating 100% of idle cloud costs.

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Ingestion & Data Lake
        A[Credit Applications Ingestion Feed] --> B[Azure Event Hubs / Stream Analytics]
        B --> C[Azure Data Lake Storage Gen2]
        C --> D[Data Contract & Validation Gatekeeper]
    end

    subgraph Feature Engineering & Calibrated Training
        D --> E[Domain Ratio Transformer: Revolving Stress & Payment Burden]
        E --> F[Gradient Boosted Decision Trees]
        F --> G[3-Fold Sigmoid Platt Probability Calibration]
        G --> H[Model Governance: ROC-AUC 0.87, Gini 0.74, Brier 0.087]
        H --> I[Serialized Artifact Registry]
    end

    subgraph Real-Time Serving & Regulatory Engines
        I --> J[Azure ML Managed Online Endpoint / Serverless API]
        J --> K{PD < 35% Risk Ceiling?}
        K -->|Approved| L[Prime Sanction Memorandum + Dynamic APR & EMI Schedule]
        K -->|Declined| M[ECOA / RBI Section 28 a Adverse Action Reasoning Engine]
    end

    subgraph Continuous MLOps Telemetry
        J --> N[Population Stability Index PSI Drift Monitor]
        J --> O[Four-Fifths Rule 80% Algorithmic Fairness Audit]
    end
```

---

## 📊 Audited Model Performance Benchmark

Evaluated on a strictly held-out $20\%$ test partition ($2,000$ credit applications):

| Evaluation Metric | Audited Score | Financial Industry Standard | Regulatory Implication |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.8716** | $\ge 0.75$ | Exceptional rank-order default discrimination |
| **Gini Coefficient** | **0.7432** | $\ge 0.60$ | Exceeds Basel III Risk Committee requirements |
| **Brier Score Loss** | **0.0878** | $< 0.12$ | High calibration fidelity for true posterior risk |
| **PR-AUC (Precision-Recall)**| **0.6158** | Baseline: $15.45\%$ | 4x precision over uncalibrated base default rate |
| **Population Stability Index**| **0.034** | $< 0.10$ | Stable; zero drift detected across live streams |
| **Disparate Impact Ratio (DIR)**| **0.949** | $\ge 0.80$ | Strict compliance with ECOA Four-Fifths Rule |

### Global Feature Attribution Breakdown
1. **Credit Bureau Score (CIBIL / FICO):** $32.6\%$
2. **Revolving Stress Index ($utilization \times credit\_deficit$):** $24.2\%$
3. **Debt-to-Income Ratio (DTI):** $9.8\%$
4. **Delinquency Severity ($late\_payments \times (1 + DTI)$):** $8.8\%$
5. **Interest Rate APR:** $8.3\%$
6. **Income-to-Loan Ratio:** $7.2\%$

---

## 💻 Enterprise Web Application Modules

The production web interface (`index.html`) is engineered with an institutional central banking aesthetic (RBI architectural watermark, left sidebar navigation, and INR ₹ currency standard):

1. **Multi-Bank Lending Institution Risk Policy Selector:** Dynamically switch underwriting mandates between **HDFC Bank** (28% risk ceiling, conservative prime), **State Bank of India (SBI)** (35% risk ceiling, mass retail), **ICICI & Axis Bank** (38% risk ceiling, agile retail), and **Digital NBFC Tier** (45% risk ceiling, dynamic risk-based pricing).
2. **0.5s High-Velocity Ingestion Terminal:** Continuous streaming application evaluation processing 120 applications/min with sub-millisecond (0.48 ms) scoring latency and one-click CSV audit export.
3. **Interactive Custom Underwriter with Visual TreeSHAP Waterfall:** Real-time financial slider evaluation decomposed through a local feature attribution waterfall (CIBIL impact, revolving stress impact, DTI impact, delinquency penalty).
4. **Official RBI Credit Sanction / Adverse Action Memo Generator:** 1-click generation of formal, watermarked credit decision memorandums downloadable as PDF or Word (.doc) files with cryptographic SHA-256 decision hashes.
5. **What-If Sensitivity & Credit Health Optimizer:** Dynamic simulation calculating the exact debt payoff or income augmentation needed to turn declined applications into approved prime loans.
6. **Enterprise Batch Credit Underwriter (CSV):** Drag-and-drop batch file uploader evaluating 30+ applications simultaneously with executive portfolio capital summaries (Total Capital, Approval Rate, Expected Loss $EL$) and CSV export.
7. **Basel III & RBI Macroeconomic Stress-Testing Simulator:** Real-time simulation of portfolio credit losses and Common Equity Tier 1 (CET1) capital adequacy under Baseline, Stagflation (+150 bps rate shock), and Crisis (+300 bps liquidity contraction) scenarios.
8. **FinTech Developer API Playground:** Interactive REST API tester for `/api/score` with live JSON request/response inspection and ready-to-use client code snippets in cURL, Python (requests), and JavaScript (fetch).

---

## 🗂️ Project Directory Structure

```text
finshield-credit-risk-mlops/
├── .github/workflows/
│   └── mlops_ci_cd.yml         # GitHub Actions automated test & Bicep validation
├── api/
│   └── score.js                # Production serverless scoring endpoint
├── artifacts/
│   ├── metrics.json            # Model benchmark evaluation records
│   └── model_pipeline.joblib   # Serialized preprocessor + calibrated pipeline
├── data/
│   ├── raw/                    # Raw synthetic credit applications (10,000 records)
│   ├── processed/              # Feature engineered data & validation reports
│   └── test/                   # Benchmark evaluation payloads
├── infra/
│   └── main.bicep              # Azure IaC (AML, ADLS Gen2, Key Vault, ACR)
├── pipelines/
│   ├── azureml_pipeline.py     # Azure ML v2 Python SDK pipeline definition
│   └── pipeline_job.yaml       # Azure ML v2 Cloud Job Manifest
├── public/
│   └── index.html              # Synchronized production web portal
├── src/
│   ├── data_validation.py      # Enterprise data contract schema & bounds auditor
│   ├── feature_engineering.py  # Domain-specific financial interaction transformers
│   ├── train.py                # Model training, 3-fold calibration & evaluation
│   ├── explainability.py       # ECOA & RBI Adverse Action reasoning generator
│   └── score.py                # Local batch and cloud scoring runtime
├── tests/
│   └── test_pipeline.py        # Automated test suite (5/5 unit tests passing)
├── index.html                  # Enterprise banking dashboard & risk terminal
├── vercel.json                 # Serverless routing and hosting configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Enterprise system documentation
```

---

## 🚀 Quickstart: Local Execution

### 1. Validate Data Contracts
```powershell
python src/data_validation.py
```

### 2. Feature Engineering & Calibrated Model Training
```powershell
python src/feature_engineering.py
python src/train.py
```

### 3. Run Automated Unit Test Suite
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

### 4. Test Local Scoring & Explainability Engine
```powershell
python src/score.py
```

---

## ☁️ Azure Cloud Infrastructure Deployment

When deploying to Microsoft Azure:
```bash
# 1. Login to Azure CLI
az login

# 2. Create Resource Group in Central India (or preferred region)
az group create --name rg-finshield-prod --location centralindia

# 3. Provision Infrastructure via Bicep (Zero-Idle Compute)
az deployment group create --resource-group rg-finshield-prod --template-file infra/main.bicep

# 4. Trigger Azure ML Training Pipeline Job
az ml job create --file pipelines/pipeline_job.yaml --resource-group rg-finshield-prod
```

---

## 🛡️ Regulatory Compliance & Fair Lending Certifications

* **Reserve Bank of India (RBI):** Master Direction on Digital Lending (2022) & Fair Practices Code for NBFCs/Banks.
* **Equal Credit Opportunity Act (ECOA / US Regulation B):** 12 CFR Part 1002 Mandated Adverse Action disclosures.
* **Federal Reserve Board SR 11-7:** Supervisory Guidance on Model Risk Management (MRM).
* **Basel Committee on Banking Supervision (BCBS):** Basel III Internal Ratings-Based (IRB) metrics for capital adequacy.

---

## ⚠️ System Limitations & Synthetic Data Disclosure

In compliance with banking data privacy regulations and Model Risk Management (MRM) standards:

1. **Synthetic Data Generation for Privacy Protection:**
   - In accordance with the **Digital Personal Data Protection (DPDP) Act 2023**, **RBI KYC/Data Localization mandates**, and **GLBA/GDPR**, no proprietary customer Personally Identifiable Information (PII) or real bank accounts are used in this repository.
   - All 10,000+ credit records were programmatically generated using multivariate statistical correlation structures (`src/generate_data.py`), accurately modeling empirical Indian retail credit parameters (CIBIL score distributions, debt-to-income ratios, revolving credit card utilization, and baseline 15.45% default rates).

2. **Absence of Alternative / Unstructured Credit Data:**
   - The current model relies on traditional bureau and capacity ratios. Real-world commercial lending increasingly integrates alternative underwriting signals (e.g., Account Aggregator bank statement analysis, GST return filings, utility bill regularity), which are not present in the synthetic dataset.

3. **Macroeconomic Feedback Loops:**
   - While the platform features a Basel III stress-testing simulator, synthetic data cannot fully model black-swan contagion effects, sudden localized regulatory moratoriums, or industry-specific systemic shocks without live production bureau feeds.

4. **Institutional Prototype Scope:**
   - FinShield is an enterprise architecture proof-of-concept designed for cloud evaluation and regulatory sandbox simulation; production deployment within a bank requires integration with core banking systems (e.g., Finacle, BaNCS) and formal sign-off from an independent Model Validation Committee.

---

## 📄 License
Distributed under the Apache 2.0 License.
