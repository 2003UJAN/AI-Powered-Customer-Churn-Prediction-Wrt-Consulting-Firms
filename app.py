import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
import os
from dotenv import load_dotenv

# --- Page Configuration ---
st.set_page_config(
    page_title="Client Churn Predictor 📈",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Gemini API Configuration ---
# Load environment variables from .env file for local development
load_dotenv()

# Configure Gemini API
# Try to get the key from Streamlit's secrets first (for deployment)
# Fall back to environment variable from .env file (for local development)
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

gemini_model = None
if api_key:
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}")
else:
    st.error("Google Gemini API key not found. Please set it in your Streamlit secrets or a local .env file.")

# --- Load Model and Assets ---
@st.cache_data
def load_assets():
    """Loads the pre-trained model and SHAP explainer."""
    model_pipeline = joblib.load('assets/model.pkl')
    preprocessor = model_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']
    explainer = shap.TreeExplainer(classifier)
    return model_pipeline, preprocessor, explainer

try:
    model, preprocessor, explainer = load_assets()
except FileNotFoundError:
    st.error("Model assets not found. Please run train_model.py to generate them.")
    st.stop()


# --- Helper Functions ---
def generate_retention_strategy(client_data, prediction_proba):
    """Generates a personalized retention strategy using the Gemini API."""
    if not gemini_model:
        return "Gemini API not configured. Cannot generate strategy."

    prompt = f"""
    As an expert business consultant specializing in client retention, you are tasked with creating a personalized retention strategy for a client who is at risk of churning.

    Client Profile:
    - Firm Size: {client_data['FirmSize'].iloc[0]}
    - Industry: {client_data['Industry'].iloc[0]}
    - Contract Tenure: {client_data['ContractTenureMonths'].iloc[0]} months
    - Projects Completed: {client_data['ProjectsCompleted'].iloc[0]}
    - Average Project Value: ${client_data['AverageProjectValue'].iloc[0]:,.2f}
    - Recent Support Tickets: {client_data['SupportTickets'].iloc[0]}
    - Client Satisfaction Score: {client_data['ClientSatisfactionScore'].iloc[0]}/5
    - Days Since Last Interaction: {client_data['LastInteractionDays'].iloc[0]}
    - Monthly Recurring Revenue (MRR): ${client_data['MonthlyRecurringRevenue'].iloc[0]:,.2f}

    Prediction:
    - The client has a {prediction_proba*100:.2f}% probability of churning.

    Your Task:
    Based on the client's profile and high churn risk, provide a concise, actionable, and personalized retention strategy. Structure your response with the following sections:
    1.  **Immediate Actions:** 2-3 specific, high-priority steps to take right now.
    2.  **Proactive Engagement Plan:** A short-term plan to re-engage the client and demonstrate value.
    3.  **Long-Term Value Proposition:** Suggestions to enhance the long-term relationship and prevent future churn.

    Keep the tone professional, empathetic, and solution-oriented.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the strategy: {e}"


def plot_shap_explanation(input_df):
    """Generates and returns a SHAP waterfall plot for a single prediction."""
    transformed_input = preprocessor.transform(input_df)
    shap_values = explainer.shap_values(transformed_input)
    feature_names = preprocessor.get_feature_names_out()
    
    explanation = shap.Explanation(
        values=shap_values[0], 
        base_values=explainer.expected_value, 
        data=transformed_input[0], 
        feature_names=feature_names
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig.tight_layout()
    return fig

# --- UI Layout ---
st.title("🤖 AI-Powered Client Churn Prediction")
st.markdown("For Consulting and Financial Services Firms")

# --- Sidebar for Data Input ---
st.sidebar.header("Client Data Input")
input_method = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

input_df = None

if input_method == "Manual Input":
    st.sidebar.subheader("Enter Client Details Manually")
    firm_size = st.sidebar.selectbox("Firm Size", ['Small (<50)', 'Medium (50-250)', 'Large (250+)', 'Enterprise (>1000)'])
    industry = st.sidebar.selectbox("Industry", ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing'])
    tenure = st.sidebar.slider("Contract Tenure (Months)", 1, 72, 12)
    projects = st.sidebar.slider("Projects Completed", 0, 100, 10)
    project_value = st.sidebar.number_input("Average Project Value ($)", min_value=1000, max_value=200000, value=25000, step=1000)
    tickets = st.sidebar.slider("Support Tickets (Last Quarter)", 0, 50, 5)
    satisfaction = st.sidebar.slider("Client Satisfaction Score (1-5)", 1, 5, 3)
    last_interaction = st.sidebar.slider("Days Since Last Interaction", 0, 365, 30)
    mrr = st.sidebar.number_input("Monthly Recurring Revenue ($)", min_value=500, max_value=50000, value=5000, step=500)

    manual_data = {
        'FirmSize': [firm_size], 'Industry': [industry], 'ContractTenureMonths': [tenure],
        'ProjectsCompleted': [projects], 'AverageProjectValue': [project_value],
        'SupportTickets': [tickets], 'ClientSatisfactionScore': [satisfaction],
        'LastInteractionDays': [last_interaction], 'MonthlyRecurringRevenue': [mrr]
    }
    input_df = pd.DataFrame(manual_data)

else: # Upload CSV
    uploaded_file = st.sidebar.file_uploader("Upload your client data CSV", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)


# --- Main Panel for Output ---
if input_df is not None:
    st.header("Prediction Results")
    
    required_cols = ['FirmSize', 'Industry', 'ContractTenureMonths', 'ProjectsCompleted', 'AverageProjectValue', 'SupportTickets', 'ClientSatisfactionScore', 'LastInteractionDays', 'MonthlyRecurringRevenue']
    
    if not all(col in input_df.columns for col in required_cols):
        st.error(f"The uploaded CSV is missing one or more required columns. Please include: {', '.join(required_cols)}")
    else:
        input_df_ordered = input_df[required_cols]

        prediction_proba = model.predict_proba(input_df_ordered)[:, 1]
        input_df['ChurnProbability'] = prediction_proba
        
        st.subheader("Client Churn Analysis")
        
        if len(input_df) > 1:
            client_index = st.selectbox("Select a client from the uploaded file to analyze:", input_df.index, format_func=lambda x: f"Client at row {x} (Prob: {input_df.loc[x, 'ChurnProbability']:.2%})")
        else:
            client_index = 0
            
        selected_client_data = input_df_ordered.iloc[[client_index]]
        selected_client_proba = input_df.loc[client_index, 'ChurnProbability']
        annual_revenue_at_risk = selected_client_data['MonthlyRecurringRevenue'].iloc[0] * 12

        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_level = "HIGH" if selected_client_proba > 0.5 else "LOW"
            st.metric(
                label="Churn Risk Probability",
                value=f"{selected_client_proba:.2%}",
                delta=f"{risk_level} RISK",
                delta_color="inverse"
            )
            st.metric(
                label="Potential Annual Revenue at Risk",
                value=f"${annual_revenue_at_risk:,.2f}"
            )

        with col2:
            st.subheader("Key Factors Influencing Prediction")
            shap_fig = plot_shap_explanation(selected_client_data)
            st.pyplot(shap_fig, use_container_width=True)

        if risk_level == "HIGH":
            st.subheader("🤖 GenAI-Powered Retention Strategy")
            if st.button("Generate Retention Plan", key=f"gen_{client_index}"):
                with st.spinner("🧠 Consulting with Gemini AI..."):
                    strategy = generate_retention_strategy(selected_client_data, selected_client_proba)
                    st.markdown(strategy)

        if len(input_df) > 1:
            st.subheader("Full Prediction Results for Uploaded Data")
            st.dataframe(input_df)
else:
    st.info("👋 Welcome! Please enter client data manually or upload a CSV file to get a churn prediction.")
    st.subheader("Model Performance (on Test Set)")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    perf_col1.metric("Accuracy", "94.50%")
    perf_col2.metric("AUC Score", "0.9134")
    perf_col3.metric("F1-Score", "0.7826")
