
# 📉 Customer Churn Prediction
### Proactive Retention Intelligence using Gradient Boosting

<p align="center">
<img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/Scikit--Learn-Classification-F7931E?style=for-the-badge&logo=scikitlearn">
<img src="https://img.shields.io/badge/Model-XGBoost-2EAD33?style=for-the-badge">
<img src="https://img.shields.io/badge/ROC--AUC-0.8554-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Platform-Jupyter-orange?style=for-the-badge&logo=jupyter">
</p>

---

## 🌟 Overview

Customer churn is one of the most significant challenges for service-based businesses. This project develops a high-performance **Supervised Machine Learning** pipeline to identify at-risk customers in the telecom sector. By analyzing behavioral patterns and account metrics, the model provides actionable insights that allow for targeted retention strategies.



---

## 📊 Dataset Overview

The study utilizes a telecom dataset containing **7,043 customer records**, capturing the lifecycle of a user through demographic and service-specific features.

* **Demographics:** Gender, Senior Citizen status, and family dynamics.
* **Account Metrics:** Tenure, Contract Type (Monthly vs. Annual), and Billing Methods.
* **Service Usage:** Fiber optic availability, Online Security, and Tech Support.
* **Target:** `Churn` (Binary: Yes/No).

---

## 🎯 Project Workflow

1.  **Data Preprocessing:** Handling null values, one-hot encoding for categorical variables, and feature scaling.
2.  **Exploratory Data Analysis (EDA):** Identifying correlations between tenure and churn rates.
3.  **Model Benchmarking:** Implementing a competitive bake-off between Logistic Regression, Random Forest, and XGBoost.
4.  **Optimization:** Using **10-fold cross-validation** to ensure model stability across different data folds.
5.  **Feature Importance:** Extracting the specific drivers of churn to inform business logic.



---

## 🧠 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **ML Framework** | Scikit-learn, XGBoost |
| **Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## 📁 Project Structure

```bash
Customer-Churn-Prediction/
├── src/
│   └── churn.py              # Implementation of ML classification pipeline
├── data/
│   └── customer_churn.csv    # Raw dataset (Telco Churn)
├── reports/
│   └── churn_report.txt      # Comprehensive analysis results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation

```


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone [https://github.com/G-Narendra/Customer-Churn-Prediction.git](https://github.com/G-Narendra/Customer-Churn-Prediction.git)
cd Customer-Churn-Prediction

```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt

```

### 3️⃣ Execute the Model

```bash
python churn.py

```

---

## 📉 Model Performance & Evaluation

The following results were achieved through 10-fold cross-validation, with **XGBoost** emerging as the champion model.

| Model | Mean ROC-AUC | Stability (Std Dev) |
| --- | --- | --- |
| **XGBoost** | **0.8523** | **± 0.0087** |
| **Random Forest** | 0.8457 | ± 0.0113 |
| **Logistic Regression** | 0.8412 | ± 0.0098 |

### **Champion Model Metrics (XGBoost)**

* **Final ROC-AUC:** 0.8554
* **Accuracy:** 80.43%

---

## 🔍 Key Churn Drivers

The analysis identifies high-risk profiles based on the following feature impacts:

* **Tenure:** New customers show a significantly higher risk of churn compared to long-term subscribers.
* **Contract Type:** Month-to-month contracts are the primary source of churn.
* **Monthly Charges:** High billing amounts without bundled security/support services correlate with attrition.
* **Service Type:** Fiber optic users exhibit higher churn, potentially indicating service quality issues.

---

## Engineering Decisions & Challenges Solved

| Challenge | Decision | Why |
|---|---|---|
| Churn is rare (imbalanced dataset) | SMOTE oversampling + class-weighted XGBoost | Naive training predicts "no churn" for everyone — SMOTE ensures the model learns churn patterns |
| Feature interactions are non-linear | XGBoost over Logistic Regression — captures feature interactions automatically | Linear models miss interactions like "high monthly charges + short tenure = high churn risk" |
| Business needs actionable interventions, not just predictions | SHAP values ranking features by churn impact | Knowing *why* a customer will churn enables targeted retention strategies, not just flagging |
| Model threshold affects precision/recall trade-off | Precision-Recall curve analysis with business-aligned threshold selection | Default 0.5 threshold is arbitrary — the optimal threshold depends on the cost of false positives vs missed churn |

## 👨‍💻 Author

**Narendra (G‑Narendra)** AI | ML | Python | Full Stack | GenAI Enthusiast

📧 [Email Me](mailto:narendragandikota2540@gmail.com) | 💼 [LinkedIn](https://linkedin.com/in/g-narendra/) | 👨‍💻 [GitHub](https://github.com/G-Narendra)

---

<p align="center">⭐ If you find this project useful, feel free to give it a star! 🚀</p>

