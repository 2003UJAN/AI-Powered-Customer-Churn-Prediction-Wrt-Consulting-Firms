import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.models import load_model
from utils import read_dataset, prepare_sequences  # prepare_sequences exists in train_models; replicate small helper if import fails

# helper to prepare tabular features (duplicate lightweight)
def prepare_tabular_features(df):
    features = [
        "tenure_months", "contract_len_months", "annual_revenue", "num_projects", "industry_risk",
        "usage_mean_12m", "usage_std_12m", "usage_last_3m_mean", "support_calls_12m", "bill_mean_12m"
    ]
    return df[features], features

@st.cache_data
def load_models(model_dir="models"):
    xgb_bundle = joblib.load(os.path.join(model_dir, "xgb_model.joblib"))
    xgb_model = xgb_bundle["model"]
    scaler = xgb_bundle["scaler"]
    features = xgb_bundle["features"]
    lstm_model = None
    if os.path.exists(os.path.join(model_dir, "lstm_model")):
        lstm_model = load_model(os.path.join(model_dir, "lstm_model"))
    metrics = {}
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return xgb_model, scaler, features, lstm_model, metrics

def shap_summary_plot(xgb_model, X, feature_names):
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X)
    fig = shap.plots.beeswarm(shap_values, show=False)
    return fig

def main():
    st.set_page_config(page_title="AI Churn - Consulting Firms", layout="wide")
    st.title("AI-Powered Customer Churn Prediction — Consulting Firms")
    st.markdown("Tabular (XGBoost) + Sequential (LSTM) ensemble with SHAP explainability.")

    model_dir = st.sidebar.text_input("Models directory", "models")
    xgb_model, scaler, feature_names, lstm_model, metrics = load_models(model_dir)

    st.sidebar.header("Metrics (test)")
    if metrics:
        st.sidebar.write("**XGBoost**")
        st.sidebar.json(metrics.get("xgb_test", {}))
        st.sidebar.write("**LSTM**")
        st.sidebar.json(metrics.get("lstm_test", {}))
        st.sidebar.write("**Ensemble**")
        st.sidebar.json(metrics.get("ensemble_test", {}))
    else:
        st.sidebar.warning("No metrics.json found in models folder.")

    st.header("Upload dataset (synthetic or your own)")
    uploaded = st.file_uploader("Upload CSV with same columns as synthetic dataset", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    else:
        if st.button("Load sample synthetic data (generated)"):
            # try to load data/synthetic_customers.csv
            sample_path = "data/synthetic_customers.csv"
            if os.path.exists(sample_path):
                df = pd.read_csv(sample_path)
            else:
                st.error("No sample data found. Please run generate_data.py first.")
                df = None
        else:
            df = None

    if df is not None:
        st.write("Dataset preview:")
        st.dataframe(df.head())

        # prepare tabular
        X_tab, features = prepare_tabular_features(df)
        X_scaled = scaler.transform(X_tab[features].values)

        # XGBoost predictions
        xgb_proba = xgb_model.predict_proba(X_scaled)[:, 1]

        # LSTM predictions if available
        lstm_proba = None
        if lstm_model is not None:
            # decode sequences column if present (strings)
            def decode_list(s):
                try:
                    return json.loads(s)
                except:
                    return s

            seqs = []
            for idx, row in df.iterrows():
                usage = decode_list(row.get("usage_seq", "[]"))
                calls = decode_list(row.get("support_calls_seq", "[]"))
                seq_arr = np.vstack([usage, calls]).T
                seqs.append(seq_arr)
            X_seq = np.stack(seqs)
            # simple normalization used in training is unknown here, but we run through model directly
            lstm_proba = lstm_model.predict(X_seq).ravel()
        else:
            st.info("LSTM model not found — ensemble will use XGBoost only.")

        if lstm_proba is not None:
            ensemble_proba = 0.5 * xgb_proba + 0.5 * lstm_proba
        else:
            ensemble_proba = xgb_proba

        df["xgb_proba"] = xgb_proba
        if lstm_proba is not None:
            df["lstm_proba"] = lstm_proba
        df["ensemble_proba"] = ensemble_proba
        df["predicted_churn"] = (df["ensemble_proba"] >= 0.5).astype(int)

        st.subheader("Predictions")
        st.dataframe(df[["customer_id", "ensemble_proba", "predicted_churn"]].sort_values("ensemble_proba", ascending=False).head(20))

        # single-customer inspector
        st.subheader("Single customer inspector")
        cust_id = st.selectbox("Pick a customer id", df["customer_id"].tolist())
        cust_row = df[df["customer_id"] == cust_id].iloc[0]
        st.write("Predicted churn probability (ensemble):", float(cust_row["ensemble_proba"]))
        st.write("Features:")
        st.json({k: float(cust_row[k]) if k in cust_row.index and pd.api.types.is_numeric_dtype(type(cust_row[k])) else cust_row.get(k, None) for k in feature_names})

        # SHAP summary (for XGBoost)
        st.subheader("SHAP explainability (XGBoost)")
        if st.button("Show SHAP summary plot"):
            with st.spinner("Computing SHAP..."):
                # shap expects 2D numpy array
                X_for_shap = X_scaled[:200]  # limit for speed
                explainer = shap.Explainer(xgb_model)
                shap_vals = explainer(X_for_shap)
                fig = shap.plots.beeswarm(shap_vals, show=False)
                st.pyplot(bbox_inches="tight")
                plt.clf()
    else:
        st.info("Upload or load sample data to get predictions.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Made for consulting firms — highlight business impact: show predicted churn rates by segment, top drivers per client using SHAP, and expected revenue at risk.")

if __name__ == "__main__":
    main()
