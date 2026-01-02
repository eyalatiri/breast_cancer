🩺 **Clinical Decision Support System in Breast Oncology**

This repository implements an integrated clinical decision support system designed to assist clinicians throughout the breast cancer patient journey, from diagnosis to treatment response prediction.
The system follows the CRISP-DM methodology and combines machine learning, clinical data, and radiomics to provide interpretable and high-performance predictions.

🧠 **Project Overview**

Breast cancer management involves multiple complex decisions under uncertainty. This project addresses these challenges by proposing six complementary modelling objectives (Objectives 1–5 and 7) that support:

-Early and reliable diagnosis

-Treatment strategy selection

-Risk stratification

-Prediction of response to neoadjuvant chemotherapy

The system is designed as a clinical decision support tool, not as a replacement for medical expertise.

📊 **Datasets Used**

The system was developed and validated using three complementary datasets:

-**Wisconsin Diagnostic Breast Cancer (WDBC)**:569 samples/30 numerical morphological features/Used for tumour classification benchmarking

-**METABRIC**:1,904 patients/Longitudinal clinical and biological data/Used for therapeutic decisions and risk modelling

-**MAMA-MIA**:277 patients/1,448 DCE-MRI images/Multi-modal (clinical + imaging)/Used for neoadjuvant treatment response (pCR) prediction

🔄 **CRISP-DM Methodology**

The project follows the six phases of CRISP-DM:

-**Business Understanding** – Clinical problem definition and objectives

-**Data Understanding** – Exploration of clinical, biological, and imaging data

-**Data Preparation** – Cleaning, normalization, feature engineering, radiomics extraction

-**Modeling** – Evaluation of multiple ML and DL models per objective

-**Evaluation** – Clinical-oriented metrics (AUC, recall, calibration)

-**Deployment** – Flask-based applications with isolated environments

🤖 **Machine Learning Models & Objectives**

**Objective 1 – Tumour Diagnostic Model (Classification)**

Objective: Distinguish between malignant and benign tumours using morphological features for early and objective diagnosis.

Models Evaluated:Softmax Regression, L2-SVM, k-Nearest Neighbors, Linear Regression (Perceptron), Deep MLP

Chosen Model: Deep Multilayer Perceptron (MLP)

Performance:

Accuracy: 98.83%

AUC-ROC: 0.9947

**Objective 2 – Therapeutic Decision Support Model**

Objective: Predict the probability of prescribing chemotherapy, hormone therapy, or radiotherapy, acting as a triage tool (YES / NO / UNCERTAIN).

Models Evaluated: MultiOutput Logistic Regression

Chosen Model: MultiOutput Logistic Regression

Performance:

AUC (Chemotherapy): 0.904

Rationale: High interpretability and calibrated probabilities, essential for clinical safety.

**Objective 3 – Intensity of Therapy Estimation Model**

Objective: Estimate a latent treatment intensity score reflecting the global burden of a therapeutic plan.

Models Evaluated:

Item Response Theory (IRT – 2PL)

Ridge Regression

k-Nearest Neighbors

Chosen Pipeline: IRT 2PL + Ridge + k-NN

Performance:

Correlation with observed outcomes: 0.508

Innovation: Creation of a latent clinical score not explicitly present in raw data.

**Objective 4 – Hormonal Resistance Risk Model**

Objective: Identify patients at high risk of relapse under hormone therapy, serving as a proxy for hormonal resistance.

Models Evaluated: XGBoost

Chosen Model: Calibrated XGBoost

Performance:

Recall: 0.763

Rationale: Prioritizes sensitivity to avoid missing high-risk patients.

**Objective 5 – Post-Treatment Relapse Risk Model**

Objective: Estimate the probability of cancer recurrence after treatment, enabling risk-based follow-up strategies.

Models Evaluated:Random Forest/XGBoost/LightGBM

Chosen Model: Stacking Ensemble with Logistic Regression meta-model

Performance:

Recall: 0.751

**Objective 6 – Pathological Complete Response (pCR) Prediction Model**

Objective: Predict before surgery whether a patient will achieve a Pathological Complete Response (pCR) after neoadjuvant chemotherapy.

Models Evaluated: ANN, XGBoost, LightGBM

Model	AUC-ROC
ANN	0.692
XGBoost	0.775
LightGBM (retained)	0.799

Chosen Model: LightGBM

Recall: 0.895

**Feature Extraction & Fusion**

-Radiomics:4,043 features extracted from DCE-MRI using PyRadiomics

-Reduced to 100 features via:Variance Threshold/ANOVA F-test/Random Forest importance

-Clinical Data: Biomarkers (ER, PR, HER2), age, tumour stage

-Final Balance: 70% radiomics / 30% clinical

🧩 **System Architecture & Deployment Strategy**
Why Two Virtual Environments?
The project integrates heterogeneous libraries with incompatible dependency requirements, particularly involving NumPy and PyRadiomics C-extensions.

Component	Constraint:
-PyRadiomics	Requires NumPy < 2.0
-TensorFlow (MLP)	Requires newer NumPy
-LightGBM / XGBoost	Compatible with modern NumPy

A single environment causes:DLL load failures/NumPy C-extension crashes/Runtime instability

-Adopted Solution: Dual Flask Architecture
🧠 **Environment 1 – Clinical & Tabular Models**

Flask application

Diagnosis, therapy decision, risk models

TensorFlow, XGBoost, LightGBM

NumPy ≥ 2.0

🧬** Environment 2 – Radiomics & pCR**

Flask application

Radiomics feature extraction + pCR prediction

PyRadiomics, SimpleITK

NumPy 1.23.x

->Each application runs on a different port and communicates via URL redirection, ensuring stability and reproducibility.

🛠 **Technologies**

Language: Python 3.9

Framework: Flask

Libraries:
Scikit-learn, Pandas, NumPy, TensorFlow, PyRadiomics, XGBoost, LightGBM, SimpleITK
