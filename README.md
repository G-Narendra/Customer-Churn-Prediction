# 📉 Customer Churn Prediction

**Predicting which customers will churn (leave) a service using gradient boosting.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E.svg)](https://scikit-learn.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🎯 Problem Statement

Predicting which customers will churn (leave) a service using gradient boosting.

---

## 📊 What I Built

ML classification pipeline for customer churn prediction using gradient boosting.

### Key Results

| Metric | Value |
|---|---|
| **Model** | XGBoost (if present) or Logistic Regression |
| **Train Size** | 70% |
| **Test Size** | 30% |
| **Evaluation** | accuracy, f1, confusion_matrix |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas |
| **ML Framework** | Scikit-Learn |
| **Model** | XGBoost (if present) or Logistic Regression |

---

## 📁 Project Structure

```
Customer-Churn-Prediction/
├── *.ipynb                          # Main notebook with full pipeline
├── ml_evaluation_utils.py           # Evaluation utilities (CV, confidence intervals)
├── README.md
└── LICENSE
```

---

## 🔧 How to Run

```bash
# Install dependencies
pip install pandas scikit-learn jupyter

# Run the notebook
jupyter notebook *.ipynb
```

---

## 🧪 Engineering Decisions

| Decision | Rationale |
|---|---|
| **XGBoost (if present) or Logistic Regression** | Chosen as baseline model for this problem type |
| **70/30 Split** | Standard split ratio for small-medium datasets |
| **Random State 2529** | Fixed random state ensures reproducibility |

---

## ⚠️ Limitations

- **Class imbalance**
- **No cross-validation**
- **No confidence intervals**

---

## ⚠️ Disclaimer

This is an educational project for learning ML concepts. It is not intended for production use.

---

*Built as part of MSc Data Science coursework — demonstrating fundamental ML pipeline.*
