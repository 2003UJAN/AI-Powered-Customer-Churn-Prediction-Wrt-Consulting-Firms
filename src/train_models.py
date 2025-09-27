import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import read_dataset, train_test_split_custom, metrics_report
from sklearn.model_selection import train_test_split
import json

def prepare_tabular(df):
    # choose tabular features (excluding sequence columns)
    features = [
        "tenure_months", "contract_len_months", "annual_revenue", "num_projects", "industry_risk",
        "usage_mean_12m", "usage_std_12m", "usage_last_3m_mean", "support_calls_12m", "bill_mean_12m"
    ]
    X = df[features].values
    y = df["churn"].values
    return X, y, features

def prepare_sequences(df, seq_len=12):
    # Use usage_seq and support_calls_seq as multivariate sequence input
    X_seq = []
    for i, row in df.iterrows():
        usage = np.array(json.loads(row["usage_seq"])) if isinstance(row["usage_seq"], str) else np.array(row["usage_seq"])
        calls = np.array(json.loads(row["support_calls_seq"])) if isinstance(row["support_calls_seq"], str) else np.array(row["support_calls_seq"])
        # stack channels: (seq_len, features)
        stacked = np.vstack([usage, calls]).T  # shape (seq_len, 2)
        X_seq.append(stacked)
    X_seq = np.stack(X_seq)  # (n_samples, seq_len, channels)
    y = df["churn"].values
    return X_seq, y

def build_lstm(input_shape):
    model = models.Sequential()
    model.add(layers.Masking(mask_value=0., input_shape=input_shape))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    df = read_dataset(args.data)

    # split
    train_df, test_df = train_test_split_custom(df, test_size=0.2)
    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    # === XGBoost (tabular) ===
    X_train, y_train, feature_names = prepare_tabular(train_df)
    X_test, y_test, _ = prepare_tabular(test_df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    xgb = XGBClassifier(n_estimators=200, max_depth=5, use_label_encoder=False, eval_metric="logloss", random_state=42)
    print("Training XGBoost...")
    xgb.fit(X_train_s, y_train)

    # save
    joblib.dump({"model": xgb, "scaler": scaler, "features": feature_names}, os.path.join(args.out_dir, "xgb_model.joblib"))

    # metrics
    xgb_proba_test = xgb.predict_proba(X_test_s)[:, 1]
    xgb_metrics = metrics_report(y_test, xgb_proba_test)
    print("XGBoost test metrics:", xgb_metrics)

    # === LSTM (sequential) ===
    Xs_train, ys_train = prepare_sequences(train_df)
    Xs_test, ys_test = prepare_sequences(test_df)
    # normalize per-channel
    # reshape to (n_samples, seq_len, channels)
    seq_shape = Xs_train.shape[1:]
    Xs_train_reshaped = Xs_train.copy()
    Xs_test_reshaped = Xs_test.copy()

    # simple channel-wise scaling
    for ch in range(Xs_train_reshaped.shape[2]):
        ch_vals = Xs_train_reshaped[:, :, ch].ravel()
        mean = ch_vals.mean()
        std = ch_vals.std() + 1e-6
        Xs_train_reshaped[:, :, ch] = (Xs_train_reshaped[:, :, ch] - mean) / std
        Xs_test_reshaped[:, :, ch] = (Xs_test_reshaped[:, :, ch] - mean) / std

    lstm = build_lstm(input_shape=seq_shape)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]
    print("Training LSTM...")
    lstm.fit(Xs_train_reshaped, ys_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)

    lstm.save(os.path.join(args.out_dir, "lstm_model"))

    lstm_proba_test = lstm.predict(Xs_test_reshaped).ravel()
    lstm_metrics = metrics_report(ys_test, lstm_proba_test)
    print("LSTM test metrics:", lstm_metrics)

    # === Simple ensemble (average probabilities) ===
    ensemble_proba = 0.5 * xgb_proba_test + 0.5 * lstm_proba_test
    ens_metrics = metrics_report(y_test, ensemble_proba)
    print("Ensemble metrics:", ens_metrics)

    # save metrics to a file
    metrics_out = {
        "xgb_test": xgb_metrics,
        "lstm_test": lstm_metrics,
        "ensemble_test": ens_metrics
    }
    import json
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("Saved models and metrics to", args.out_dir)
