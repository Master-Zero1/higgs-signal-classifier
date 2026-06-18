<div align="center">

# Higgs Signal Classification using XGBoost and SHAP

[![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square&logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange?style=flat-square)](https://xgboost.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.0-red?style=flat-square)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple?style=flat-square)](https://shap.readthedocs.io)

</div>

---

## Overview

This project investigates the classification of Higgs boson signal events from background noise using machine learning on the HIGGS dataset. This study reproduces and extends the work of Baldi et al. (2014) by comparing Logistic Regression and modern XGBoost models across different feature groups, then performing hyperparameter optimization, analyzing feature importance using SHAP, and evaluating the effect of training dataset size through learning curve analysis.

---

## Research Questions
1. How does modern XGBoost perform compared to Baldi et al. (2014) BDT, NN, and DNN -> How much classical ML improved?
2. Which physics features actually drive the classification?
3. How much information is contained in low-level detector features versus high-level physics engineered features?
4. How much training data is required before performance begins to saturate?

---

## Dataset

**Source:** UCI Machine Learning Repository - HIGGS Dataset

**Dataset Characteristics**
- **Total Events - (~ 11,000,000)**
- **Features - 28**
- **Target - Binary Classification (Signal vs Background)**
- **Low-Level Features - 21**
- **High-Level Features - 7**

### Signal vs. Background Count
![Signal vs background count](plots/signal%20vs%20background%20count.png)

The first 21 features correspond to detector-level measurements, while the remaining 7 are physics-derived variables constructed from combinations of low-level measurements. 

### Features correlations (low-level features and high level features)
![Feature correlation](plots/feature_correlation%20matrix.png)

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
An XGBoost model with the best parameter was trained on progressively larger subsets:

| Step | 10K | 50K | 100K | 250K | 500K | 1M | 2.5M | 5M |

---

## Results

**Validation Performance**

| Model | 21 features | 7 features | 28 features |
|-------|--------|---------|-----------------|
| Logistic Regression | 0.5944 | 0.6451 | 0.6831 |
| XGBoost (default) | 0.7265 | 0.7899 | 0.8240 |
| XGBoost (tuned) | **0.8007** | **0.7962** | **0.8547** |

**Final Test Performance**

| Model | Test AUC Score |
|-------|--------|
| XGBoost (tuned) 21 features | 0.8005 |
| XGBoost (tuned) 7 features | 0.7955 |
| XGBoost (tuned) 28 features | **0.8548** |

**SHAP Feature Importance**

**Top 5 Most Important Features (28 features)**

| Rank | Feature | MeanAbsSHAP |
|-------|--------|--------|
| 1 | m_bb | 0.605438 |
| 2 | m_wbb | 0.301321 |
| 3 | jet 1 pt | 0.293371 |
| 4 | m_jjj | 0.282922 |
| 5 | m_jlv | 0.275762 |

These results align with known physics expectations regarding Higgs boson decay signatures, indicating that invariant mass metrics provide the strongest geometric cuts for classification.

### SHAP Summary Plot - (28 Features)

![SHAP Summary](plots/SHAP%20Summary%20Plot%20-%20Tuned%20XGBoost%20(28%20features).png)

### SHAP Feature Importance Ranking - (28 Features)

![SHAP Importance](plots/SHAP%20Feature%20Importance%20Ranking%20(28%20features).png)

### SHAP Dependence plot

**Most important feature overall: m_bb**

![Dependence Plot](plots/Most%20important%20feature%20dependence%20plot.png)

### SHAP Summary Plot - (7 Features)

![SHAP Summary](plots/SHAP%20Summary%20Plot%20-%20High%20level%20features.png)

### SHAP Feature Importance Ranking - (7 Features)

![SHAP Importance](plots/SHAP%20Feature%20Importance%20Ranking%20(7%20Features).png)

**Learning Curve - Results**

| Training Size | Validation AUC | Best Iterations | Training Time (sec) | AUC Gain |
|-------|-------|-------|-------|-------|
| 10K | 0.782333 | 55 | 1.823531 | NaN |
| 50K | 0.805421 | 108 | 1.475551 | 0.023087 |
| 100K | 0.813080 | 190 | 2.207243 | 0.007659 |
| 250K | 0.821936 | 413 | 4.575684 | 0.008856 |
| 500K | 0.828882 | 577 | 7.442126 | 0.006945 |
| 1M | 0.834993 | 1034 | 16.674511 | 0.006111 |
| 2.5M | 0.843019 | 2261 | 64.083215 | 0.008026 |
| 5M | 0.848819 | 3526 | 194.734479 | 0.005801 |

### Learning Curve - Tuned XGBoost

![Learning curve](plots/Learning%20Curve%20-%20Tuned%20XGBoost.png)

### Training Time vs Dataset Size

![Training Time vs Dataset Size](plots/Training%20Time%20vs%20Dataset%20Size%20-%20learning%20curve.png)

### Early stopping Iteration vs Dataset Size

![Early stopping Iteration vs Dataset Size](plots/Early-Stopping%20Iteration%20vs%20Dataset%20Size%20-%20learning%20curve.png)

**Key Findings**

1. High-level features are highly informative
The physics - engineered features provide substantial discriminative power and outperform low-level detector measurements when used alone.

2. Combining feature sets produces the best performance.
The highest performance was achieved using all 28 features
**Final Test AUC: **0.8548

3. Hyperparameter tuning significantly improves performance
| Feature | Default AUC | Tuned AUC |
|---------|------------:|----------:|
| 21 Features |	0.7265 | 0.8005 |
| 7 Features | 0.7899 |	0.7955 |
| 28 Features |	0.8240 | 0.8548 |

4. Validation and test performance are consistent
| Feature Set | Validation AUC | Test AUC |
|---------|------------:|----------:|
| 21 Features |	0.8007 | 0.8005 |
| 7 Features | 0.7962 |	0.7955 |
| 28 Features |	0.8547 | 0.8548 |

---

## Technologies Used
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Scikit-Learn**
- **XGBoost**
- **SHAP**

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_eda_plots.ipynb
│   ├── 03_ml_pipeline.ipynb
|   ├── 04_ml_models.ipynb
│   ├── 05_shap_analysis.ipynb
│   └── 06_learning_curve.ipynb
│
├── data/
│   ├── raw
|   ├── processed
│   └── output
│
├── plots/
│   ├── high_level_features/
│   │   ├── m_bb.png
│   │   ├── m_jjj.png
│   │   ├── m_jj.png
│   │   ├── m_jlv.png
│   │   ├── m_lv.png
│   │   ├── m_wbb.png
│   │   └── m_wwbb.png
│   │
│   ├── low_level_features/
│   │   ├── jet 1 b-tag.png
│   │   ├── jet 1 eta.png
│   │   ├── jet 1 phi.png
│   │   ├── jet 1 pt.png
│   │   ├── jet 2 b-tag.png
│   │   ├── jet 2 eta.png
│   │   ├── jet 2_phi.png
│   │   ├── jet 2 pt.png
│   │   ├── jet 3 b-tag.png
│   │   ├── jet 3 eta.png
│   │   ├── jet 3 phi.png
│   │   ├── jet 3 pt.png
│   │   ├── jet 4 b-tag.png
│   │   ├── jet 4 eta.png
│   │   ├── jet 4 phi.png
│   │   ├── jet 4 pt.png
│   │   ├── lepton eta.png
│   │   ├── lepton phi.png
│   │   ├── lepton pt.png
│   │   ├── missing energy magnitude.png
│   │   └── missing energy phi.png
|   |
│   ├── signal vs background count.png
|   ├── feature_correlation matrix.png
│   ├── SHAP Summary Plot - Tuned XGBoost (28 features).png
│   ├── SHAP Feature Importance Ranking (28 features).png
|   ├── SHAP Summary Plot - High level features.png
|   ├── SHAP Feature Importance Ranking (7 Features).png
│   ├── Learning Curve - Tuned XGBoost.png
|   ├── Training Time vs Dataset Size - learning curve.png
│   └── Early-Stopping Iteration vs Dataset Size - learning curve.png
│
├── models/
│   ├── logistic_regression_21.pkl
│   ├── logistic_regression_7.pkl
│   ├── logistic_regression_28.pkl
│   ├── xgb_21_default.pkl
│   ├── xgb_7_default.pkl
│   ├── xgb_28_default.pkl
│   ├── final_xgb_21_tuned.pkl
│   ├── final_xgb_7_tuned.pkl
│   └── final_xgb_28_tuned.pkl
│
├── requirements.txt
└── README.md

```

---

## References
Baldi, P., Sadowski, P., & Whiteson, D. (2014).
Searching for Exotic Particles in High-Energy Physics with Deep Learning.

Nature Communications, 5, 4308.

UCI Machine Learning Repository - HIGGS Dataset