import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained LightGBM model and scaler
model = joblib.load("lgbm_churn_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define column order (same as training data)
feature_columns = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
    'Total Spend', 'Last Interaction', 'Subscription Type_Premium', 
    'Subscription Type_Standard', 'Contract Length_Monthly', 'Contract Length_Quarterly'
]

# Streamlit App UI
st.title("Customer Churn Prediction App")
st.write("Fill in the details below to predict customer churn.")

# Create a form for user inputa
with st.form("churn_form"):
    age = st.number_input("Age", 18, 100, 30)
    gender = st.radio("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (months)", 0, 60, 12)
    usage_frequency = st.number_input("Usage Frequency (per month)", 0, 30, 10)
    support_calls = st.number_input("Number of Support Calls", 0, 20, 2)
    payment_delay = st.number_input("Payment Delay (days)", 0, 60, 5)
    total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
    last_interaction = st.number_input("Days Since Last Interaction", 0, 365, 30)

    subscription_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract_length = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Yearly"])

    # Submit button
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Encode categorical variables
    gender_encoded = 1 if gender == "Male" else 0

    subscription_type_standard = 1 if subscription_type == "Standard" else 0
    subscription_type_premium = 1 if subscription_type == "Premium" else 0

    contract_length_monthly = 1 if contract_length == "Monthly" else 0
    contract_length_quarterly = 1 if contract_length == "Quarterly" else 0

    # Create a DataFrame for the model
    input_data = pd.DataFrame([[age, gender_encoded, tenure, usage_frequency, support_calls, 
                                payment_delay, total_spend, last_interaction, 
                                subscription_type_premium, subscription_type_standard, 
                                contract_length_monthly, contract_length_quarterly]], 
                              columns=feature_columns)

    # Scale numerical features
    num_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display prediction result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error("This customer is predicted to churn. 🚨")
    else:
        st.success("This customer is not likely to churn. ✅")

st.write("**Note:** 1 = Churn, 0 = No Churn")
