import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib
import os

# Load data
df = pd.read_csv('assets/synthetic_data.csv')

# Define features (X) and target (y)
X = df.drop(['ClientID', 'Churn'], axis=1)
y = df['Churn']

# Identify categorical and numerical features
categorical_features = ['FirmSize', 'Industry']
numerical_features = X.select_dtypes(include=np.number).columns.tolist()

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create the full model pipeline with XGBoost
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Model Performance on Test Set:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Save the trained model pipeline
joblib.dump(model, 'assets/model.pkl')

print("\n✅ Model trained and saved to assets/model.pkl")
