import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np # Required for the mock model

# --- Mock Model for UI Demo Mode ---
class MockModelPipeline:
    """
    A fake model that mimics the real one's behavior when the real model fails to load.
    This allows the UI to function without crashing. Predictions generated are NOT real.
    """
    def predict_proba(self, df):
        """
        Generates plausible-looking predictions based on a simple rule (NPS score).
        """
        # A lower NPS score will result in a higher churn probability for demonstration.
        nps_scores = df['nps_score']
        
        # Start with a base low probability for all clients.
        base_prob = np.full(len(df), 0.15) 
        
        # Increase probability for clients with an NPS score below 7.
        # This is a vectorized operation: it applies the logic to all rows at once.
        churn_prob = np.where(nps_scores < 7, base_prob + (7 - nps_scores) * 0.1, base_prob)
        churn_prob = np.clip(churn_prob, 0.05, 0.95) # Keep probability within a realistic range.
        
        # The real model outputs probabilities for both classes (0 and 1).
        # We must return an array of shape (n_samples, 2).
        # Column 1: P(not churn), Column 2: P(churn).
        return np.vstack([1 - churn_prob, churn_prob]).T

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Client Churn Predictor",
    page_icon="💼",
    layout="wide",
)

# --- Custom CSS for a Modern Look ---
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
INDUSTRY_CATEGORIES = ['Tech', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Logistics']
COMPANY_SIZE_CATEGORIES = ['<50', '50-250', '250-1000', '1000+']
SERVICE_LEVEL_CATEGORIES = ['Basic', 'Premium', 'Enterprise']
PAYMENT_HISTORY_CATEGORIES = ['On-time', 'Late', 'Consistently Late']

# --- Asset Loading ---
@st.cache_resource
def load_model(model_path):
    """
    Tries to load the real model. If it fails due to a version mismatch error,
    it returns a functional Mock Model instead, activating Demo Mode.
    """
    global is_demo_mode
    if not os.path.exists(model_path):
        st.error(f"Model file not found at `{model_path}`. Please run `python src/model_pipeline.py`.")
        return None
    try:
        # Attempt to load the real model file.
        pipeline = joblib.load(model_path)
        is_demo_mode = False # If successful, we are NOT in demo mode.
        return pipeline
    except (AttributeError, TypeError) as e:
        # **THIS IS THE FIX**: If the specific version error occurs, load the mock model.
        st.warning(
            "⚠️ **DEMO MODE ACTIVATED**\n\n"
            f"The real model at `{model_path}` could not be loaded due to a library version mismatch (`{e}`).\n\n"
            "The app is now running with a **mock model**. All predictions are for demonstration purposes only and are **not real**."
        )
        is_demo_mode = True # Set the flag to indicate we are in demo mode.
        return MockModelPipeline()

@st.cache_data
def load_data(data_path):
    """Loads the client dataset."""
    if not os.path.exists(data_path):
        st.error(f"Data file not found at `{data_path}`. Please run data generation first.")
        return None
    return pd.read_csv(data_path)

# --- Load Data and Model ---
is_demo_mode = False # Global flag to track if we're in demo mode.
model_pipeline = load_model(MODEL_PATH)
df_original = load_data(DATA_PATH)

# --- Main Application ---
st.title("💼 AI-Powered Client Churn Predictor")
if is_demo_mode:
    st.error("**WARNING: You are in Demo Mode. The predictions shown below are not from the real AI model.**")

# --- Graceful Exit if Files are Missing ---
if model_pipeline is None or df_original is None:
    st.warning("One or more required files could not be loaded. The application cannot proceed.")
    st.stop()

# --- Predictions for Batch Data ---
if 'predictions_made' not in st.session_state or st.session_state.get('is_demo_mode', -1) != is_demo_mode:
    df_predict = df_original.copy()
    churn_probabilities = model_pipeline.predict_proba(df_predict)[:, 1]
    df_predict['churn_probability'] = churn_probabilities
    st.session_state.df_predictions = df_predict
    st.session_state.predictions_made = True
    st.session_state.is_demo_mode = is_demo_mode

df_predictions = st.session_state.df_predictions

# --- UI Layout ---
col1, col2 = st.columns([0.6, 0.4])

# --- Column 1: Batch Client Analysis ---
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
        st.markdown("**Select a high-risk client to generate a tailored retention plan.**")
        if not high_risk_clients.empty:
            selected_client_name = st.selectbox("Select Client:", high_risk_clients['company_name'].tolist())
            if st.button(f"Generate Plan for {selected_client_name}"):
                client_data = high_risk_clients[high_risk_clients['company_name'] == selected_client_name].iloc[0]
                with st.spinner("🧠 Generating strategy..."):
                    # This part still needs the gemini_integration.py and a .env file to work
                    from src.gemini_integration import get_retention_strategy
                    strategy = get_retention_strategy(client_data.to_dict(), client_data['churn_probability'])
                    st.markdown(strategy)
        else:
             st.info("No high-risk clients to analyze.")

# --- Column 2: Manual Data Entry ---
with col2:
    st.header("🔮 Predict for a Single Client")
    with st.form(key='single_client_form'):
        st.markdown("Enter a client's details to get a real-time churn prediction.")

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
                 with st.spinner("🧠 Generating strategy..."):
                    from src.gemini_integration import get_retention_strategy
                    manual_data_dict = {k: v[0] for k, v in manual_data.items()}
                    manual_data_dict['company_name'] = "This Manually Entered Client"
                    strategy = get_retention_strategy(manual_data_dict, churn_prob)
                    st.markdown(strategy)
