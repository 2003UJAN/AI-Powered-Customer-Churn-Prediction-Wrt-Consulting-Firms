import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys

# Add the 'src' directory to the Python path to allow for module imports
# This is a common practice in structured Python projects.
# It ensures that we can import our custom preprocessing module.
try:
    from data_preprocessing import get_preprocessing_pipeline
except ImportError:
    # Handle the case where the script is run from a different directory
    sys.path.append(os.path.dirname(__file__))
    from data_preprocessing import get_preprocessing_pipeline

# --- Define File Paths ---
DATA_PATH = 'data/consulting_churn_data.csv'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'churn_model_xgb.joblib')

def train_model_pipeline():
    """
    Executes the full model training and serialization pipeline.
    
    1. Loads the raw dataset.
    2. Splits the data into training and testing sets.
    3. Defines a scikit-learn pipeline that combines preprocessing and the XGBoost model.
    4. Trains the pipeline on the training data.
    5. Evaluates the trained pipeline on the test data.
    6. Saves the final, fitted pipeline object to a file for later use in the app.
    """
    
    # --- 1. Load Data ---
    print("Step 1/5: Loading dataset...")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"-> Successfully loaded {DATA_PATH} with {len(df)} records.")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at '{DATA_PATH}'.")
        print("Please run the data generation notebook/script first to create the dataset.")
        return

    # --- 2. Split Data ---
    print("Step 2/5: Splitting data into training and testing sets...")
    # 'X' contains all the features for prediction. We drop the target ('churn')
    # and non-predictive identifiers.
    X = df.drop(['churn', 'client_id', 'company_name'], axis=1)
    # 'y' is the target variable we want to predict.
    y = df['churn']

    # Splitting the data into 80% for training and 20% for testing.
    # 'stratify=y' is crucial for classification tasks, especially with imbalanced data.
    # It ensures that the proportion of churned vs. non-churned clients is the
    # same in both the training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("-> Data split complete.")

    # --- 3. Define the scikit-learn Pipeline ---
    print("Step 3/5: Defining the model pipeline...")
    
    # 
    # The Pipeline chains together multiple steps. The output of each step is the input
    # to the next. This is powerful because it bundles all logic into a single object.
    pipeline = Pipeline(steps=[
        # Step A: Preprocessing
        # This uses the ColumnTransformer we defined in `data_preprocessing.py`
        # to handle categorical features.
        ('preprocessor', get_preprocessing_pipeline()),
        
        # Step B: Classification Model
        # This is the XGBoost model that will be trained on the preprocessed data.
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic', # For binary classification (Churn vs. No Churn)
            eval_metric='logloss',       # Evaluation metric to monitor during training
            use_label_encoder=False,     # Modern XGBoost requirement
            random_state=42              # For reproducibility
        ))
    ])
    print("-> Pipeline defined successfully.")

    # --- 4. Train the Pipeline ---
    print("Step 4/5: Training the model... (This may take a moment)")
    # When we call .fit() on the pipeline, it sequentially fits and transforms the data
    # at each step, finally training the XGBoost classifier on the fully preprocessed data.
    pipeline.fit(X_train, y_train)
    print("-> Model training complete.")

    # --- 5. Evaluate and Save the Pipeline ---
    print("Step 5/5: Evaluating and saving the model...")
    # Make predictions on the unseen test data.
    y_pred = pipeline.predict(X_test)
    
    # Calculate performance metrics.
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'])
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy on Test Set: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("--------------------------------\n")
    
    # Ensure the 'models' directory exists before saving.
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save the entire trained pipeline object to a file.
    # joblib is efficient for saving scikit-learn objects with large numpy arrays.
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Model pipeline successfully trained and saved to '{MODEL_PATH}'")

# --- Main Execution Block ---
# This ensures the code inside only runs when the script is executed directly.
if __name__ == "__main__":
    train_model_pipeline()
