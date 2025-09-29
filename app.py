import streamlit as st
import pandas as pd
import numpy as np
import joblib
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
load_dotenv()
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
    """Loads the pre-trained model."""
    model_pipeline = joblib.load('assets/model.pkl')
    return model_pipeline

try:
    model = load_assets()
except FileNotFoundError:
    st.error("Model assets not found. Please run train_model.py to generate them.")
    st.stop()


# --- Helper Functions ---
def generate_gemini_explanation(client_data, prediction_proba):
    """Generates a natural language explanation for the churn prediction using Gemini."""
    if not gemini_model:
        return "Gemini API not configured. Cannot generate explanation."

    # Create a summary of the client's data for the prompt
    profile_summary = "\n".join([f"- **{col.replace('_', ' ')}:** {client_data[col].iloc[0]}" for col in client_data.columns])
    
    risk_level = "HIGH" if prediction_proba > 0.5 else "LOW"

    prompt = f"""
    You are an expert business analyst interpreting a machine learning model's prediction. Your task is to provide a clear, concise, and business-friendly explanation for a client churn prediction. Do not mention the model itself (e.g., "the model predicted"). Instead, speak authoritatively based on the data.

    **Client Data Profile:**
    {profile_summary}

    **Prediction Outcome:**
    - **Churn Risk:** {risk_level} ({prediction_proba:.2%})

    **Your Task:**
    Based on the client's data, explain **why** they are considered a {risk_level} churn risk.
    1.  Start with a clear, one-sentence summary of the situation.
    2.  In a bulleted list, highlight the top 3-4 key factors that are most likely influencing this prediction. For each factor, briefly explain its business impact (e.g., "A low satisfaction score of 2/5 suggests they are unhappy with our service," or "A long tenure of 48 months indicates a historically strong relationship, which is a positive sign.").
    3.  Conclude with a final sentence that frames the situation positively or negatively, depending on the risk.

    Keep the language simple, direct, and actionable for a business manager.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the explanation: {e}"

def generate_retention_strategy(client_data, prediction_proba):
    """Generates a personalized retention strategy using the Gemini API."""
    if not gemini_model:
        return "Gemini API not configured. Cannot generate strategy."

    prompt = f"""
    As an expert business consultant, create a personalized retention strategy for a client at risk of churning.

    **Client Profile & Prediction:**
    - The client has a {prediction_proba*100:.2f}% probability of churning.
    - Firm Size: {client_data['FirmSize'].iloc[0]}
    - Industry: {client_data['Industry'].iloc[0]}
    - Contract Tenure: {client_data['ContractTenureMonths'].iloc[0]} months
    - Projects Completed: {client_data['ProjectsCompleted'].iloc[0]}
    - Average Project Value: ${client_data['AverageProjectValue'].iloc[0]:,.2f}
    - Recent Support Tickets: {client_data['SupportTickets'].iloc[0]}
    - Client Satisfaction Score: {client_data['ClientSatisfactionScore'].iloc[0]}/5
    - Days Since Last Interaction: {client_data['LastInteractionDays'].iloc[0]}
    - Monthly Recurring Revenue (MRR): ${client_data['MonthlyRecurringRevenue'].iloc[0]:,.2f}

    **Your Task:**
    Provide a concise, actionable retention strategy with these sections:
    1.  **Immediate Actions:** 2-3 specific steps to take now.
    2.  **Proactive Engagement Plan:** A short-term plan to demonstrate value.
    3.  **Long-Term Value Proposition:** Suggestions to enhance the long-term relationship.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the strategy: {e}"

# --- UI Layout ---
st.title("🤖 AI-Powered Client Churn Prediction")
st.markdown("For Consulting and Financial Services Firms")

# --- Sidebar ---
st.sidebar.header("Client Data Input")
input_method = st.sidebar.radio("Choose input method", ["Manual Input", "Upload CSV"])

input_df = None
if input_method == "Manual Input":
    # (Manual input fields remain the same as before)
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

else:
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
        
        client_index = 0
        if len(input_df) > 1:
            client_index = st.selectbox("Select a client to analyze:", input_df.index, format_func=lambda x: f"Client at row {x} (Prob: {input_df.loc[x, 'ChurnProbability']:.2%})")
            
        selected_client_data = input_df_ordered.iloc[[client_index]]
        selected_client_proba = input_df.loc[client_index, 'ChurnProbability']
        annual_revenue_at_risk = selected_client_data['MonthlyRecurringRevenue'].iloc[0] * 12
        risk_level = "HIGH" if selected_client_proba > 0.5 else "LOW"

        # --- Display Metrics and Gemini Explanation ---
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Churn Risk Probability", value=f"{selected_client_proba:.2%}", delta=f"{risk_level} RISK", delta_color="inverse")
            st.metric(label="Potential Annual Revenue at Risk", value=f"${annual_revenue_at_risk:,.2f}")

        with col2:
            st.subheader("Prediction Explanation")
            with st.spinner("🤖 Gemini is analyzing the prediction..."):
                explanation = generate_gemini_explanation(selected_client_data, selected_client_proba)
                st.markdown(explanation)

        # --- GenAI Retention Strategy ---
        if risk_level == "HIGH":
            st.subheader("🤖 GenAI-Powered Retention Strategy")
            if st.button("Generate Retention Plan", key=f"gen_{client_index}"):
                with st.spinner("🧠 Consulting with Gemini AI..."):
                    strategy = generate_retention_strategy(selected_client_data, selected_client_proba)
                    st.markdown(strategy)

        if len(input_df) > 1:
            st.subheader("Full Prediction Results")
            st.dataframe(input_df)
else:
    st.info("👋 Welcome! Please enter client data or upload a CSV to get a churn prediction.")
    # (Model Performance metrics remain the same)
