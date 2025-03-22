# **📉 Customer Churn Prediction using Machine Learning**

*A machine learning model to predict customer churn based on historical data.*

## 🌟 **Overview**
This project implements a **Customer Churn Prediction model** using **Supervised Machine Learning** techniques. The goal is to identify customers who are likely to churn, helping businesses take proactive retention measures.

## 📊 **Dataset Overview**
The dataset used is a **customer churn dataset** from a telecom company. It contains **7,043 customer records** with the following key features:
- **Demographics**: Gender, Senior Citizen, Partner, Dependents
- **Account Info**: Tenure, Contract Type, Payment Method, Monthly & Total Charges
- **Services Availed**: Internet Service, Online Security, Tech Support, Streaming Services
- **Target Variable**: **Churn (Yes/No)**

## 🎯 **Project Workflow**
✅ **Data Preprocessing** – Handling missing values, encoding categorical variables, feature scaling.  
✅ **Feature Engineering** – Extracting meaningful insights to improve model predictions.  
✅ **Model Training & Evaluation** – Comparing multiple machine learning models.  
✅ **Model Interpretation** – Analyzing feature importance to understand churn patterns.  

## 🛠️ **Tech Stack**
🔹 **Programming Language:** Python  
🔹 **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
🔹 **Model Type:** Classification (Logistic Regression, Random Forest, XGBoost)  
🔹 **Development Environment:** Jupyter Notebook / Python Script  

## 📂 **Project Structure**
```
Customer-Churn-Prediction/
├── churn.py                  # Python script with model implementation
├── customer_churn.csv         # Dataset used for training/testing
├── churn_intro.txt            # Dataset overview
├── churn_report.txt           # Detailed project report
├── requirements.txt           # Dependencies for the project
├── README.md                  # Project documentation
└── data/                      # Additional data (if applicable)
```

## 🚀 **Installation & Setup**
1️⃣ **Clone the Repository**  
```sh
git clone https://github.com/G-Narendra/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```
2️⃣ **Install Dependencies**  
```sh
pip install -r requirements.txt
```
3️⃣ **Run the Model**  
```sh
python churn.py
```

## 📉 **Model Performance & Evaluation**
Three models were evaluated using **10-fold cross-validation (ROC-AUC metric):**
| Model                 | Mean ROC-AUC |
|----------------------|-------------|
| Logistic Regression | 0.8412 ± 0.0098 |
| Random Forest       | 0.8457 ± 0.0113 |
| **XGBoost (Best)**  | **0.8523 ± 0.0087** |

### **Final Model Evaluation (XGBoost)**
| Metric      | Score  |
|------------|--------|
| ROC-AUC    | 0.8554 |
| Accuracy   | 80.43% |

### **Classification Report (XGBoost)**
```
              precision    recall  f1-score   support

           0       0.85      0.87      0.86      1033
           1       0.65      0.61      0.63       374

    accuracy                           0.80      1407
   macro avg       0.75      0.74      0.74      1407
weighted avg       0.80      0.80      0.80      1407
```

## 🔍 **Key Features Influencing Churn**
Feature importance analysis using XGBoost revealed that the **top predictors of churn** are:
- **Tenure** (longer tenure customers are less likely to churn)
- **Monthly Charges** (higher charges increase churn probability)
- **Total Charges**
- **Contract Type (Two-Year Contracts)**
- **Internet Service (Fiber Optic)**

## 🤝 **Contributions**
💡 Open to improvements! Feel free to:
1. Fork the repo  
2. Create a new branch (`feature-branch`)  
3. Make changes & submit a PR  


## 📩 **Connect with Me**
📧 **Email:** [narendragandikota2540@gmail.com](mailto:narendragandikota2540@gmail.com)  
🌐 **Portfolio:** [G-Narendra Portfolio](https://g-narendra-portfolio.vercel.app/)  
💼 **LinkedIn:** [G-Narendra](https://linkedin.com/in/g-narendra/)  
👨‍💻 **GitHub:** [G-Narendra](https://github.com/G-Narendra)  

⭐ **If you find this project useful, drop a star!** 🚀
