import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# It's good practice to import your own modules with an alias
from src.gemini_integration import get_retention_strategy

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Client Churn Predictor",
    page_icon="💼",
    layout="wide",
)

# --- Custom CSS for a cleaner look ---
st.markdown("""
<style>
.stMetric {
    border-radius: 10px;
    background-color: #f0f2f6;
    padding: 15px;
}
.stButton>button {
    width: 100%;
    border-radius: 10px;
}
.stDataFrame {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# --- File Paths & Constants ---
DATA_PATH = 'data/consulting_churn_data.csv'
MODEL_PATH = 'models/churn_model_xgb.joblib'
# These categories must match the ones the model was trained on
INDUSTRY_CATEGORIES = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Logistics']
COMPANY_SIZE_CATEGORIES = ['<50', '50-250', '250-1000', '1000+']
SERVICE_LEVEL_CATEGORIES = ['Basic', 'Premium', 'Enterprise']
PAYMENT_HISTORY_CATEGORIES = ['On-time', 'Late', 'Consistently Late']


# --- Asset Caching with Error Handling ---
@st.cache_resource
def load_model(model_path):
    """Load the trained model pipeline, with specific error handling."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please run `src/model_pipeline.py`.")
        return None
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except AttributeError as e:
        st.error(
            f"An AttributeError occurred: {e}\n"
            "This error usually means the model was saved with a different library version (e.g., scikit-learn) than what's currently installed. "
            "To fix this, please delete the existing model file at 'models/churn_model_xgb.joblib' and retrain it by running `python src/model_pipeline.py` in your terminal."
        )
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

@st.cache_data
def load_data(data_path):
    """Load the client dataset if it exists."""
    if not os.path.exists(data_path):
        st.error(f"Data file not found at `{data_path}`. Please run data generation first.")
        return None
    return pd.read_csv(data_path)

# --- Load Data and Model ---
model_pipeline = load_model(MODEL_PATH)
df_original = load_data(DATA_PATH)

# --- Main Application ---
st.title("💼 AI-Powered Client Churn Predictor")
st.markdown("An interactive tool to predict client churn, analyze risk factors, and generate retention strategies.")

# --- Graceful Exit if Files are Missing ---
if model_pipeline is None or df_original is None:
    st.warning("One or more required files are missing. The application cannot proceed.")
    st.markdown("""
        **Please follow these steps in your terminal:**
        1.  **Generate Data:** Run `notebooks/1-data-generation.ipynb`.
        2.  **Train Model:** Run `python src/model_pipeline.py`.
        3.  **Refresh this page.**
    """)
    st.stop()

# --- Predictions for Batch Data (in session state) ---
if 'predictions_made' not in st.session_state:
    df_predict = df_original.copy()
    churn_probabilities = model_pipeline.predict_proba(df_predict)[:, 1]
    df_predict['churn_probability'] = churn_probabilities
    st.session_state.df_predictions = df_predict
    st.session_state.predictions_made = True

df_predictions = st.session_state.df_predictions

# --- UI Layout (Two Columns) ---
col1, col2 = st.columns([0.6, 0.4])

# --- Column 1: Batch Prediction and Analysis ---
with col1:
    st.header("👥 Batch Client Analysis")

    st.sidebar.title("Dashboard Controls")
    confidence_threshold = st.sidebar.slider(
        "High-Risk Threshold", 0.5, 1.0, 0.75, 0.05,
        help="Clients with a churn probability above this value will be flagged."
    )
    selected_industry = st.sidebar.selectbox("Filter by Industry", ['All'] + INDUSTRY_CATEGORIES)

    filtered_df = df_predictions
    if selected_industry != 'All':
        filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
    high_risk_clients = filtered_df[filtered_df['churn_probability'] >= confidence_threshold]

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Clients", f"{len(filtered_df):,}")
    m2.metric("High-Risk Clients", f"{len(high_risk_clients):,}")
    m3.metric("Avg. Churn Risk", f"{filtered_df['churn_probability'].mean():.1%}")

    tab1, tab2 = st.tabs(["High-Risk Watchlist", "AI Retention Strategies"])

    with tab1:
        st.markdown(f"**Clients with Churn Probability > {confidence_threshold:.0%}**")
        if not high_risk_clients.empty:
            st.dataframe(
                high_risk_clients[['company_name', 'industry', 'contract_value', 'nps_score', 'churn_probability']].sort_values('churn_probability', ascending=False),
                use_container_width=True,
                column_config={
                    "churn_probability": st.column_config.ProgressColumn("Churn Probability", format="%.2f", min_value=0, max_value=1),
                    "contract_value": st.column_config.NumberColumn("Contract Value", format="$%d")
                }
            )
        else:
            st.success("🎉 No clients are currently above the high-risk threshold.")

    with tab2:
        st.markdown("**Select a high-risk client to generate a tailored retention plan using Gemini AI.**")
        if not high_risk_clients.empty:
            selected_client_name = st.selectbox("Select Client:", high_risk_clients['company_name'].tolist())
            if st.button(f"Generate Plan for {selected_client_name}"):
                client_data = high_risk_clients[high_risk_clients['company_name'] == selected_client_name].iloc[0]
                with st.spinner("🧠 Gemini is crafting a strategy..."):
                    strategy = get_retention_strategy(client_data.to_dict(), client_data['churn_probability'])
                    st.markdown(strategy)
        else:
             st.info("No high-risk clients to analyze.")

# --- Column 2: Manual Data Entry and Single Prediction ---
with col2:
    st.header("🔮 Predict for a Single Client")
    with st.form(key='single_client_form'):
        st.markdown("Enter a client's details below to get a real-time churn prediction.")

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            industry = st.selectbox("Industry", INDUSTRY_CATEGORIES)
            company_size = st.selectbox("Company Size", COMPANY_SIZE_CATEGORIES)
            service_level = st.selectbox("Service Level", SERVICE_LEVEL_CATEGORIES)
            payment_history = st.selectbox("Payment History", PAYMENT_HISTORY_CATEGORIES)
        with form_col2:
            contract_value = st.number_input("Contract Value ($)", 1000, 1000000, 50000, 1000)
            contract_tenure = st.slider("Contract Tenure (Months)", 1, 72, 24)
            nps_score = st.slider("NPS Score (1-10)", 1, 10, 8)
            avg_project_rating = st.slider("Avg. Project Rating (2-5)", 2.0, 5.0, 4.0, 0.1)
        
        projects_completed = st.slider("Projects Completed", 1, 100, 10)
        communication_freq = st.slider("Monthly Communications", 1, 50, 15)
        
        submit_button = st.form_submit_button(label='Predict Churn Risk')

    if submit_button:
        manual_data = {
            'industry': [industry], 'company_size': [company_size], 'service_level': [service_level],
            'payment_history': [payment_history], 'contract_value': [contract_value],
            'contract_tenure_months': [contract_tenure], 'projects_completed': [projects_completed],
            'avg_project_rating': [avg_project_rating], 'communication_freq_monthly': [communication_freq],
            'nps_score': [nps_score]
        }
        manual_df = pd.DataFrame(manual_data)

        with st.spinner("Analyzing client profile..."):
            churn_prob = model_pipeline.predict_proba(manual_df)[0][1]
            st.subheader("Prediction Result")
            
            color = "red" if churn_prob > 0.5 else "green"
            st.markdown(f"**Churn Probability:** <span style='color:{color}; font-size: 24px;'>{churn_prob:.1%}</span>", unsafe_allow_html=True)
            st.progress(churn_prob)

            if st.button("Generate AI Retention Plan for this Profile"):
                 with st.spinner("🧠 Gemini is on the case..."):
                    manual_data_dict = {k: v[0] for k, v in manual_data.items()}
                    manual_data_dict['company_name'] = "This Manually Entered Client"
                    strategy = get_retention_strategy(manual_data_dict, churn_prob)
                    st.markdown(strategy)
