<div align="center>

# Higgs Signal Classification using XGBoost and SHAP

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-red?style=flat-square)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)](https://shap.readthedocs.io)

</div>

---

## Overview

This project investigate the classification of Higgs boson signal events from background noise using machine learning on the HIGGS dataset. This study reproduces and extends the experimental setup of Baldi et al. (2014) by comparing Logistic Regression and modern XGBoost models across different feature groups, then performing hyperparameter optimization, analyzing feature importance using SHAP, and evaluating the effect of training dataset size through learning curve analysis.

---

## Research Questions
1. How does modern XGBoost perform compared to Baldi at el. (2014) BDT, NN, and DNN -> How much classical ML improved?
2. Which physics features actually drive the classification?
3. How much information is contained in low-level detector features vesus high-level physics engineered features?
4. How much trainin data is required before performance begins to saturate?

---

## Dataset

**Source:** UCI Machine Learning Repository -> HIGGS Dataset

**Dataset Characteristics**
- **Total Events - (~ 11,000,000)**
- **Features - 28**
- **Target - Binary Classification (Signal vs Background)**
- **Low - Level Features - 21**
- **High - Level Features - 7**

The first 21 features correspond to detector-level measurements, while the remaining 7 are physics-derived variables constructed from combinations of low-level measurements. 

---

## Methodology

**Baseline Models**
The following models were trained on three feature subsets:
- **21 Low - Level features**
- **7 High - Level features**
- **All 28 Features**

**Models:**
- **Logistic Regression**
- **XGBoost**

**Hyperparameter Tuning**
RandomizedSearchCV was used to optimize:
- **max_depth**
- **learning_rate**
- **n_estimators**
- **subsample**
- **colsample_bytree**
- **min_child_weight**
- **reg_alpha**
- **reg_lambda**

**Best Parameters:**
```
max_depth        = 8
learning_rate    = 0.1
subsample        = 0.9
colsample_bytree = 0.7
min_child_weight = 3
gamma            = 0
reg_alpha        = 0.1
reg_lambda       = 1
n_estimators     = 6194
```

**Explainability**

SHAP (SHapley Additive exPlanations) was used to:
- **Rank feature importance**
- **Interpret model predictions**
- **Analyze feature effects**

**Learning Curve**
A XGBoost model with the best parameter was trained on progressively larger subsets:

| Step |
|------|
| 10K |
| 50K |
| 100K |
| 250K |
| 500K |
| 1M |
| 2.5M |
| 5M |

---

## Results

**Validation Performance**

| Model | 21 features | 7 features | 28 features |
|-------|--------|---------|-----------------|
| Logistic Regression | 0.5944 | 0.6451 | 0.6831 |
| XGBoost (default) | 0.7265 | 0.763 | 0.8240 |
| XGBoost (tuned) | **0.8007** | **0.7962** | **0.8547** |

**Final Test Performance**

| Model | Test AUC Score |
|-------|--------|
| XGBoost (tuned) 21 features | 0.8005 |
| XGBoost (tuned) 7 features | 0.7955 |
| XGBoost (tuned) 28 features | **0.8548** |

**SHAP Feature Importance**

### SHAP Summary Plot - (28 Features)

![SHAP Summary](plots/SHAP%20Summary%20Plot%20-%20Tuned%20XGBoost%20(28%20features).png)

### SHAP Summary Plot - (7 Features)

![SHAP Summary](plots/SHAP%20Summary%20Plot%20-%20High%20level%20features.png)