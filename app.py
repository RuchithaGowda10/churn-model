import streamlit as st
import requests
import openai
import os
from dotenv import load_dotenv

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="🔍 Customer Churn Predictor", layout="centered")

# 🌍 Load environment variables
load_dotenv()

# Optional: debug line (you can remove this after testing)
# st.warning(f"ML_ENDPOINT: {os.getenv('AZURE_ML_ENDPOINT')}")

# UI
st.title("🔍 Customer Churn Prediction App")
st.write("Enter customer details below:")

# Input form
credit_score = st.number_input("Credit Score", 300, 900, 720)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])
gender_val = 1 if gender == "Male" else 0

age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (Years)", 0, 10, 6)
balance = st.number_input("Account Balance", 0.0, 500000.0, 35000.50)
products = st.selectbox("Number of Products", [1, 2, 3, 4])
credit_card = st.selectbox("Has Credit Card", [1, 0])
active = st.selectbox("Is Active Member", [1, 0])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 55000.0)

# Azure configs
ml_endpoint = os.getenv("AZURE_ML_ENDPOINT")
ml_token = os.getenv("AZURE_ML_TOKEN")

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_BASE")
openai.api_version = "2025-01-01-preview"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

# GPT-4 Explanation
def get_explanation(credit_score, age, balance, products, credit_card, active, salary):
    prompt = f"""
The following customer was evaluated for churn risk. Provide a detailed explanation of each parameter and how it may contribute to churn.

- Credit Score: {credit_score}
- Age: {age}
- Balance: {balance}
- Number of Products: {products}
- Has Credit Card: {"Yes" if credit_card == 1 else "No"}
- Is Active Member: {"Yes" if active == 1 else "No"}
- Estimated Salary: {salary}

Explain clearly in plain business terms for a non-technical audience.
"""
    try:
        response = openai.ChatCompletion.create(
            engine="gpt-explainer",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"⚠️ GPT error: {str(e)}"

# Prediction logic
if st.button("Predict Churn"):
    st.info("🔄 Sending data to Azure ML model...")

    input_payload = {
        "input_data": {
            "columns": [
                "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography", "Gender",
                "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
            ],
            "index": [0],
            "data": [[
                1, 10001, "Smith", credit_score, geography, gender_val, age,
                tenure, balance, products, credit_card, active, salary
            ]]
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ml_token}"
    }

    try:
        response = requests.post(ml_endpoint, headers=headers, json=input_payload)
        result = response.json()

        if isinstance(result, list):
            prediction = result[0]
        elif isinstance(result, dict) and 'result' in result:
            prediction = result["result"][0]
        else:
            prediction = None

        if prediction == 1:
            st.error("⚠️ This customer is likely to churn.")
        elif prediction == 0:
            st.success("✅ This customer is likely to stay.")
        else:
            st.warning("⚠️ Unable to determine prediction result.")

        with st.spinner("🧠 Explaining the prediction..."):
            explanation = get_explanation(credit_score, age, balance, products, credit_card, active, salary)
            st.markdown("### 💡 GPT-4 Explanation")
            st.info(explanation)

    except Exception as e:
        st.error(f"❌ Azure ML error: {e}")