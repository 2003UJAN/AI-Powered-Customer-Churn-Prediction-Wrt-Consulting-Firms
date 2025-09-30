# src/app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from src.genai.gemini_client import GeminiClient
import ast

st.set_page_config(page_title="Churn + GenAI Retention", layout="wide")

@st.cache_resource
def load_xgb():
    pipe = joblib.load('models/xgb_pipeline.pkl')
    return pipe['model'], pipe['ohe']

@st.cache_resource
def load_lstm():
    return tf.keras.models.load_model('models/lstm_model.h5')

st.title("AI-Powered Churn Prediction — with Gemini retention suggestions")

uploaded = st.file_uploader("Upload synthetic_churn.csv (or use the example)", type=['csv'])
if uploaded is None:
    st.info("No file uploaded — using example in data/synthetic_churn.csv")
    df = pd.read_csv('data/synthetic_churn.csv')
else:
    df = pd.read_csv(uploaded)

st.dataframe(df.head())

model_xgb, ohe = load_xgb()
model_lstm = load_lstm()

# Quick feature prepare
def prepare_single(row):
    num_cols = ['tenure_months','avg_monthly_spend','num_products','days_since_last_contact']
    cat_cols = ['contract_type']
    num = np.array([row[c] for c in num_cols]).reshape(1,-1)
    cat = ohe.transform([[row['contract_type']]])
    X = np.hstack([num, cat])
    return X

st.sidebar.header("Select customer id for drilldown")
cid = st.sidebar.number_input("customer_id", min_value=int(df.customer_id.min()), max_value=int(df.customer_id.max()), value=int(df.customer_id.min()))
cust_row = df[df.customer_id==cid].iloc[0]
st.sidebar.write(cust_row.to_frame())

# Predict churn prob
X_single = prepare_single(cust_row)
prob = model_xgb.predict_proba(X_single)[0][1]
st.metric("Churn probability (XGBoost)", f"{prob:.2%}")

# LSTM sequence prediction
seq = np.array(list(map(float, cust_row['purchase_seq'].split(',')))).reshape(1, -1, 1)
lstm_prob = float(model_lstm.predict(seq).ravel()[0])
st.metric("Churn probability (LSTM)", f"{lstm_prob:.2%}")

# Combine
combined = 0.6*prob + 0.4*lstm_prob
st.header("Combined churn risk")
st.write(f"Combined score: {combined:.2%}")

# Gemini integration
st.header("Personalized retention strategy (GenAI)")
st.write("Enter optional business constraints (tone, offer types, limit budget)")
constraints = st.text_input("Constraints (e.g., 'formal tone; offer: 10% discount; prefer advisory services')", "")

if st.button("Generate retention strategy using Gemini"):
    with st.spinner("Generating..."):
        prompt = f"""
Customer snapshot:
- customer_id: {int(cust_row.customer_id)}
- tenure_months: {cust_row.tenure_months}
- avg_monthly_spend: {cust_row.avg_monthly_spend}
- num_products: {cust_row.num_products}
- days_since_last_contact: {cust_row.days_since_last_contact}
- contract_type: {cust_row.contract_type}
- churn_score: {combined:.3f}

Constraints: {constraints}

Task:
1) Suggest 3 personalized retention actions (short bullet list).
2) For each action give: expected impact (% churn reduction estimate), cost estimate, and an example message to send to the customer.
3) Provide a one-paragraph executive summary suitable for a manager.

Answer in clear bullet points.
"""
        # instantiate Gemini client (ensure authentication env set)
        g = GeminiClient()
        out = g.generate_text(prompt)
        st.markdown(out)
        # also provide a short downloadable report
        st.download_button("Download full recommendation", out, file_name=f"retention_customer_{cid}.txt")

st.write("---")
st.markdown("**Notes:** Gemini integration requires proper authentication (API key or ADC). See README.")
