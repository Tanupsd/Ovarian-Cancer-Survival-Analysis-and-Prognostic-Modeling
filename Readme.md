# Ovarian Cancer Survival Analysis Using Machine Learning and Deep Learning

An end-to-end survival analysis framework that predicts ovarian cancer patient survival using clinical data from **The Cancer Genome Atlas (TCGA-OV)**. This project compares traditional statistical survival models with machine learning and deep learning approaches to identify the most effective model for patient risk prediction.

---

## Overview

Ovarian cancer is among the leading causes of cancer-related deaths in women due to its late-stage diagnosis and high recurrence rate. Accurate survival prediction can assist clinicians in identifying high-risk patients and supporting personalized treatment planning.

This project implements a complete survival analysis pipeline consisting of data preprocessing, feature engineering, model training, evaluation, and visualization. Four different survival prediction models are implemented and compared using standard survival analysis metrics.

---

## Features

* Clinical data preprocessing and cleaning
* Missing value imputation
* Label encoding of categorical features
* Feature normalization using Min-Max Scaling
* SMOTE-based data balancing
* 80:20 Train-Test Split
* Survival prediction using multiple statistical and machine learning models
* Automatic model saving
* Model evaluation using Concordance Index (C-index)
* Confusion matrices and performance metrics
* Performance comparison visualizations

---

## Dataset

The project uses the **TCGA-OV (The Cancer Genome Atlas – Ovarian Cancer)** clinical dataset obtained from the **Genomic Data Commons (GDC) Data Portal**.

### Dataset Includes

* Patient demographic information
* Clinical characteristics
* Survival status (Alive / Dead)
* Survival time
* Multiple clinical attributes used for prognosis prediction

---

## Models Implemented

### Kaplan-Meier Estimator

A non-parametric statistical model used to estimate patient survival probabilities over time. This model serves as the baseline for comparison.

### Cox Proportional Hazards Model

A semi-parametric regression model that estimates the effect of clinical variables on patient survival using hazard ratios.

### Random Survival Forest

An ensemble-based survival learning algorithm capable of modeling complex non-linear relationships between clinical variables.

### DeepSurv

A deep neural network implemented in PyTorch that learns patient risk scores using the **Cox Partial Likelihood Loss**, enabling non-linear survival prediction.

---

## Project Workflow

```text
Clinical Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Categorical Encoding
        │
        ▼
Feature Normalization
        │
        ▼
Train-Test Split (80:20)
        │
        ├──────────────┐
        │              │
        ▼              ▼
 Training Set      Testing Set
        │
        ▼
Model Training
├── Kaplan-Meier
├── Cox Proportional Hazards
├── Random Survival Forest
└── DeepSurv
        │
        ▼
Model Evaluation
        │
        ▼
Performance Comparison
```

---

## Project Structure

```text
Ovarian-Cancer-Survival-Analysis/
│
├── data/
│   ├── clinical_data.csv
│   ├── preprocessed_data.csv
│   ├── encoded_data.csv
│   ├── normalized_data.csv
│   └── balanced_data.csv
│
├── models/
│   ├── deepsurv_model.pth
│   └── random_forest_survival_model.joblib
│
├── results/
│   ├── kaplan_meier_curve.png
│   ├── model_prediction_accuracy.png
│   ├── model_metrics_comparison.png
│   ├── Kaplan-Meier_confusion_matrix.png
│   ├── Cox Proportional Hazards_confusion_matrix.png
│   ├── Random Forest Survival_confusion_matrix.png
│   └── DeepSurv_confusion_matrix.png
│
├── ts.py
├── pre.py
├── modelload.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Scikit-survival
* Lifelines
* PyTorch
* Imbalanced-learn (SMOTE)
* Matplotlib
* Joblib

---

## Installation

Clone the repository

```bash
git clone https://github.com/Tanupsd/Ovarian-Cancer-Survival-Analysis-and-Prognostic-Modeling.git

cd Ovarian-Cancer-Survival-Analysis
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Step 1: Train the Models

```bash
python ts.py
```

This script performs:

* Data preprocessing
* Feature encoding
* Feature normalization
* Dataset balancing
* Train-test splitting
* Training of all survival models
* Saving trained models

---

### Step 2: Evaluate the Models

```bash
python pre.py
```

This generates:

* Concordance Index (C-index)
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrices
* Performance comparison graphs

---

## Experimental Results

The models were evaluated using the **Concordance Index (C-index)**, which measures the agreement between predicted and observed survival outcomes.

| Model                    |    C-Index |
| ------------------------ | ---------: |
| Kaplan-Meier Estimator   | **50.00%** |
| Cox Proportional Hazards | **60.17%** |
| DeepSurv                 | **61.88%** |
| Random Survival Forest   | **85.78%** |

### Summary

* **Random Survival Forest** achieved the highest predictive performance with a **C-index of 85.78%**, demonstrating superior ability to capture complex relationships among clinical variables.
* **DeepSurv** achieved a **61.88%** C-index using a neural network trained with the Cox Partial Likelihood objective.
* **Cox Proportional Hazards** produced competitive performance while maintaining high interpretability.
* **Kaplan-Meier** served as the baseline survival estimation method.

---

## Results

The repository includes the following visualizations:

* Kaplan-Meier Survival Curve
* Model Prediction Accuracy (C-index)
* Confusion Matrix for Kaplan-Meier
* Confusion Matrix for Cox Proportional Hazards
* Confusion Matrix for Random Survival Forest
* Confusion Matrix for DeepSurv

## Plublished to:


## 📖 Associated Publication

This repository contains the implementation of the methods presented in the following publication:

**Ovarian Cancer Survival Analysis and Prognostic Modeling**

**Authors:** Tanushree Prasad, *et al.*

**Conference:** 2025 International Conference on Decision Aid Sciences and Applications (DASA)

**Publisher:** IEEE

**DOI:** `10.1109/DASA68193.2025.11499142`

**IEEE Xplore:** https://ieeexplore.ieee.org/document/11499142

If you use this repository for academic research, please consider citing the associated publication.

