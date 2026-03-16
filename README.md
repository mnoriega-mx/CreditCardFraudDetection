# Credit Card Fraud Detection

**IS525E Data Science for Business — Group Project**
**Business Domain:** Finance
**Topic:** Risk Assessment & Fraud Detection

---

## Team members

- [Fedotova Tetiana](https://github.com/mesopotania) (20250781)
- [Freund Janna](https://github.com/jannafr) (20250782)
- [Antonio Gutierrez Mireles](https://github.com/tonogutierrez) (20250309)
- [Noriega Chapa Mauricio](https://github.com/mnoriega-mx) (20250312)
- [Ramirez Avila Gael](https://github.com/gaelramav-debug) (20250525)

---

## Project Overview

This project applies data science techniques to detect fraudulent credit card transactions. Given the high cost of undetected fraud and the business need for real-time risk assessment, we build and evaluate machine learning models capable of identifying anomalous transactions with high precision and recall.

---

## Dataset

**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
**File:** `creditcard.csv` (not tracked in git due to 144MB size — download and place in `data/`)

| Feature | Description |
|---------|-------------|
| `Time` | Seconds elapsed between transaction and first transaction |
| `V1–V28` | PCA-transformed features (anonymized for confidentiality) |
| `Amount` | Transaction amount in euros |
| `Class` | Target: `0` = legitimate, `1` = fraud |

**Size:** 284,807 transactions
**Class imbalance:** ~0.17% fraudulent transactions

---

## Project Structure

```
CreditCardFraudDetection/
├── data/                  # Dataset (not tracked in git)
│   └── creditcard.csv
├── notebooks/             # Jupyter notebooks (one per phase)
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── reports/               # Final report and visualizations
├── requirements.txt
└── README.md
```

---

## Project Steps

1. **EDA** — Explore class distribution, feature correlations, transaction amount/time patterns
2. **Data Cleaning & Preprocessing** — Handle imbalance (SMOTE/undersampling), scale features
3. **Modeling** — Train Logistic Regression, Random Forest, XGBoost; compare approaches
4. **Evaluation** — Precision, Recall, F1, AUC-ROC; business impact analysis
5. **Report** — Insights, methodology justification, business recommendations

---

## Requirements

```bash
pip install -r requirements.txt
```

Then place `creditcard.csv` in the `data/` folder and run notebooks in order.
