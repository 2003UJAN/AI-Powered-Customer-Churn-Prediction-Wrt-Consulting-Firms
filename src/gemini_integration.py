import os
import google.generativeai as genai
from dotenv import load_dotenv

def get_retention_strategy(client_data: dict, churn_prob: float) -> str:
    """
    Generates a client retention strategy using the Google Gemini API.

    This function constructs a detailed prompt based on a client's data, sends it
    to the Gemini model, and returns a formatted, actionable retention plan. It
    handles API key loading and error management.

    Args:
        client_data (dict): A dictionary containing the data of a single client.
                            Expected keys include 'company_name', 'industry', etc.
        churn_prob (float): The model's predicted churn probability for the client.

    Returns:
        str: A formatted string containing the AI-generated retention plan. If an
             error occurs (e.g., missing API key), it returns a user-friendly
             error message.
    """
    # --- 1. Load API Key ---
    # Load environment variables from a .env file into the environment
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    # Gracefully handle the case where the API key is not found.
    if not api_key:
        error_message = """
        **Error: Gemini API Key Not Found**

        To use the AI Strategy Generator, please follow these steps:
        1.  Create a file named `.env` in the root directory of this project.
        2.  Add the following line to the file, replacing `YOUR_API_KEY` with your actual key from Google AI Studio:
            ```
            GEMINI_API_KEY="YOUR_API_KEY"
            ```
        3.  Save the file and restart the Streamlit application.
        """
        return error_message

    # --- 2. Configure Gemini Model ---
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        return f"**Error:** Failed to configure the Gemini model. Please check your API key. Details: {e}"


    # --- 3. Prompt Engineering ---
    # This is the most crucial part. We give the AI a role, context, a clear data profile,
    # and a strict mandate for the output format.
    prompt = f"""
    **Role:** You are a senior Partner at a prestigious management consulting firm, renowned for your expertise in client relationship management and turning around at-risk accounts.

    **Context:** Our internal predictive model has flagged a client with a high probability of churn. You must analyze their profile and draft an immediate, actionable retention plan for the account team. Your tone should be professional, strategic, and concise.

    **Client Data Profile:**
    - **Company Name:** {client_data.get('company_name', 'N/A')}
    - **Industry:** {client_data.get('industry', 'N/A')}
    - **Contract Value:** ${client_data.get('contract_value', 0):,}
    - **Tenure with Firm:** {client_data.get('contract_tenure_months', 0)} months
    - **Recent Net Promoter Score (NPS):** {client_data.get('nps_score', 0)}/10
    - **Average Project Satisfaction Rating:** {client_data.get('avg_project_rating', 0):.1f}/5.0
    - **Monthly Communication Frequency:** {client_data.get('communication_freq_monthly', 0)} contacts

    **Model Prediction:**
    - **Predicted Churn Probability:** {churn_prob:.1%}

    **Your Mandate:**
    Based *only* on the data provided, generate the following two sections in markdown format:

    ### 1. Primary Churn Drivers
    A brief, bulleted analysis of the most likely reasons this client is dissatisfied. Infer the drivers directly from the data provided (e.g., low NPS, poor project ratings, lack of communication).

    ### 2. Strategic Retention Plan
    A numbered list of 3-4 specific, immediate actions the account team must take. The advice must be practical and tailored to the client's profile.
    """
    
    # --- 4. Call the API and Return the Response ---
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Catch potential API errors (e.g., network issues, content filtering)
        return f"**Error:** An issue occurred while contacting the Gemini API. Please try again. Details: {e}"

