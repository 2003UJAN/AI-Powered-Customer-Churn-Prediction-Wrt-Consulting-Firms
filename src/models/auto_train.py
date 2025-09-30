import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.utils import to_categorical
import numpy as np
import joblib

# --- GitHub/Local-safe path fix ---
if "__file__" in globals():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
else:
    ROOT_DIR = os.getcwd()
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data.make_dataset import get_dataset, split_dataset


def train_models(save_dir="models"):
    # Load dataset
    df = get_dataset()
    train_df, val_df, test_df = split_dataset(df)

    X_train, y_train = train_df.drop("churn", axis=1), train_df["churn"]
    X_val, y_val = val_df.drop("churn", axis=1), val_df["churn"]
    X_test, y_test = test_df.drop("churn", axis=1), test_df["churn"]

    results = {}
    os.makedirs(os.path.join(ROOT_DIR, save_dir), exist_ok=True)

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    results["RandomForest"] = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
        "f1": f1_score(y_test, preds),
    }
    joblib.dump(rf, os.path.join(ROOT_DIR, save_dir, "random_forest.pkl"))

    # --- XGBoost ---
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)
    results["XGBoost"] = {
        "accuracy": accuracy_score(y_test, preds),
        "auc": roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1]),
        "f1": f1_score(y_test, preds),
    }
    joblib.dump(xgb, os.path.join(ROOT_DIR, save_dir, "xgboost.pkl"))

    # --- Simple LSTM ---
    X_train_seq = np.expand_dims(X_train.values, axis=1)
    X_val_seq = np.expand_dims(X_val.values, axis=1)
    X_test_seq = np.expand_dims(X_test.values, axis=1)

    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)

    model = Sequential()
    model.add(LSTM(64, input_shape=(1, X_train.shape[1])))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        X_train_seq,
        y_train_cat,
        validation_data=(X_val_seq, y_val_cat),
        epochs=5,
        batch_size=32,
        verbose=0,
    )
    _, acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
    results["LSTM"] = {"accuracy": acc}

    model.save(os.path.join(ROOT_DIR, save_dir, "lstm.h5"))

    print("✅ Training results:", results)
    return results


if __name__ == "__main__":
    train_models()
