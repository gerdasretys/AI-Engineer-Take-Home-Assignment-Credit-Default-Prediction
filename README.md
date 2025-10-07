# Credit Risk Modeling Project

## Overview
This project focuses on predicting creditworthiness (good/bad borrower classification) using a dataset of 100,000 observations and 21 features. The goal was to build a machine learning model to classify customers based on financial and demographic indicators and interpret the key factors driving credit risk.

---

## 1. Data Exploration and Understanding
**Dataset shape:** (100,000, 21)  
**Target variable:** `GB` (0 = good borrower, 1 = bad borrower)  
**Class balance:** 71.5% good, 28.5% bad  

No missing values or duplicate rows were found.  
All features were numeric (integer-encoded categorical and continuous variables).

**Key feature groups:**
- **Numerical:** `CreditDuration`, `CreditAmount`, `Age`
- **Categorical:** `Balance`, `PaymentHistory`, `Purpose`, `JobDuration`, `EffortRate`, etc.

---

## 2. Feature Engineering
To enhance model interpretability and capture financial relationships, the following engineered features were created:

- **`TotalAssets`** – combination of asset-related features.  
- **`assets_to_credit`** – ratio of total assets to credit amount.  
- **`effort_per_credit`** – effort rate normalized by credit duration or amount.

After feature engineering, the dataset included **6 numeric** and **17 categorical** columns.

---

## 3. Data Preprocessing
We used a **scikit-learn ColumnTransformer pipeline** for consistent preprocessing:

- **Numerical features:** Standardized via `StandardScaler`.  
- **Categorical features:** One-hot encoded via `OneHotEncoder(handle_unknown='ignore')`.

Train-test split:  
- Train: 80,000 samples (80%)  
- Test: 20,000 samples (20%)  
- Class distribution remained consistent across both sets.

---

## 4. Modeling Approach
We trained and evaluated two baseline models:
1. **Logistic Regression** – for interpretability and a linear baseline.  
2. **Random Forest Classifier** – to capture non-linear relationships and feature interactions.

Both were wrapped in pipelines with preprocessing to ensure consistent handling of categorical and numerical features.

Hyperparameter tuning was **intentionally skipped** to focus on baseline comparison and feature importance analysis.

---

## 5. Model Evaluation

| Metric | Logistic Regression | Random Forest |
|--------|----------------------|----------------|
| Accuracy | 0.686 | 0.679 |
| Precision | 0.466 | 0.457 |
| Recall | 0.698 | 0.678 |
| F1-score | 0.559 | 0.546 |
| ROC-AUC | 0.751 | 0.741 |

Both models performed comparably, with **Logistic Regression slightly outperforming** in terms of accuracy and AUC.

### Confusion Matrix (Logistic Regression)
```
[[9732 4561]
 [1722 3985]]
```
This shows the model correctly identifies most “good” borrowers but has moderate false positives for “bad” borrowers — a common tradeoff in credit risk modeling.

---

## 6. Feature Importance Analysis

### Random Forest (Gini-based)
Top predictive features:
| Feature | Importance |
|----------|-------------|
| Balance_4 | 0.241 |
| Balance_1 | 0.141 |
| CreditDuration | 0.105 |
| PaymentHistory_4 | 0.047 |
| TotalAssets | 0.036 |
| OtherAssets_1 | 0.030 |
| assets_to_credit | 0.025 |
| CreditAmount | 0.024 |
| effort_per_credit | 0.023 |
| Age | 0.020 |

Interpretation:
- Higher **balance categories (Balance_4)** indicate increased credit risk.
- **Credit duration** and **credit amount** also strongly influence repayment likelihood.
- **Asset-related ratios** contribute meaningfully to default risk assessment.

### Permutation Importance
Permutation-based results confirmed the dominance of a few features:
| Feature | Mean Importance |
|----------|----------------|
| CreditDuration | 0.097 |
| CreditAmount | 0.015 |
| Age | 0.013 |
| effort_per_credit | 0.011 |

Interpretation:  
The duration and size of credit remain the most predictive features, followed by customer age and repayment effort. This aligns with standard financial domain intuition.

---

## 7. Insights and Recommendations
- **Model performance:** Both baseline models show moderate predictive power (AUC ≈ 0.75), which is reasonable for a first pass without tuning or advanced feature selection.
- **Interpretation:** Loan duration, balance, and credit amount drive credit risk. Asset and payment history also play key roles.
- **Next steps:**  
  - Apply SMOTE or class weighting to address imbalance.  
  - Perform hyperparameter tuning (GridSearchCV/Optuna).  
  - Evaluate advanced models (XGBoost, LightGBM).  
  - Implement model calibration for probability-based risk scoring.

---

## 8. Use of AI Assistance
Some parts of this project (notably, documentation formatting and debugging support for feature importance computation) were developed with the assistance of **OpenAI’s ChatGPT (GPT-5)**.  
AI assistance was used to:
- Structure the modeling pipeline.
- Generate code for exploratory data analysis and model evaluation.
- Summarize and interpret the output results.
- Draft this README documentation in a concise, professional format.

All decisions regarding feature selection, model choice, and evaluation interpretation were made by Gerdas Retys.

---

## 9. Conclusion
This project establishes a solid baseline for credit risk prediction. Logistic Regression provides interpretability, while Random Forest captures non-linear effects. Future improvements can focus on tuning, ensemble methods, and better handling of class imbalance for enhanced predictive accuracy.

---

**Author:** Gerdas Retys
**Date:** October 2025  
**Tools:** Python, scikit-learn, pandas, matplotlib, seaborn  
