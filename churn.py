import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import classification_report

# Load the data
df = pd.read_csv('customer_churn.csv')

# Data preprocessing
# Drop customerID as it's not needed for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill missing TotalCharges with the median
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Convert 'Churn' to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Separate features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Define categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing pipelines for both numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Column transformer to apply appropriate transformations to columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Evaluate models
results = {}
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='roc_auc')
    results[name] = scores
    print(f"{name}: Mean ROC-AUC = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# Choose the best model (based on cross-validated ROC-AUC score)
best_model_name = max(results, key=lambda k: np.mean(results[k]))
print(f"Best model: {best_model_name}")

# Train the best model on the entire training set
best_model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', models[best_model_name])])
best_model.fit(X_train, y_train)

# Evaluate the best model on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

print("Test set evaluation:")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# Feature importance for the best model if it's tree-based
if best_model_name == 'Random Forest' or best_model_name == 'XGBoost':
    feature_importances = best_model.named_steps['classifier'].feature_importances_
    # Get feature names after preprocessing
    feature_names = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names(categorical_cols)
    feature_names = np.append(numerical_cols, feature_names)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
