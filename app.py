import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")

st.markdown(
"""
This application predicts whether a **telecom customer is likely to churn**.

You can:
- 🔹 Predict churn for a **single customer**
- 🔹 Upload a **dataset for batch predictions**
"""
)

# Expected model columns
expected_columns = [
"SeniorCitizen",
"tenure",
"MonthlyCharges",
"TotalCharges",
"gender_Male",
"Partner_Yes",
"Dependents_Yes",
"PhoneService_Yes",
"MultipleLines_Yes",
"InternetService_Fiber optic",
"InternetService_No",
"OnlineSecurity_Yes",
"OnlineBackup_Yes",
"DeviceProtection_Yes",
"TechSupport_Yes",
"StreamingTV_Yes",
"StreamingMovies_Yes",
"Contract_One year",
"Contract_Two year",
"PaperlessBilling_Yes",
"PaymentMethod_Credit card (automatic)",
"PaymentMethod_Electronic check",
"PaymentMethod_Mailed check"
]

# Tabs
tab1, tab2 = st.tabs(["🔹 Manual Prediction", "📂 Batch Prediction"])

# ---------------- MANUAL PREDICTION ---------------- #

with tab1:

    st.subheader("Enter Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0)
        monthly_charges = st.number_input("Monthly Charges")

        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    # automatic/default values
    gender = 0
    partner = 0
    dependents = 0
    phone_service = 1
    multiple_lines = 0
    online_security = 0
    online_backup = 0
    device_protection = 0
    tech_support = 0
    streaming_tv = 0
    streaming_movies = 0
    paperless_billing = 1

    total_charges = tenure * monthly_charges

    features = [
        1 if senior_citizen == "Yes" else 0,
        tenure,
        monthly_charges,
        total_charges,
        gender,
        partner,
        dependents,
        phone_service,
        multiple_lines,
        1 if internet_service == "Fiber optic" else 0,
        1 if internet_service == "No" else 0,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        1 if contract == "One year" else 0,
        1 if contract == "Two year" else 0,
        paperless_billing,
        1 if payment_method == "Credit card (automatic)" else 0,
        1 if payment_method == "Electronic check" else 0,
        1 if payment_method == "Mailed check" else 0
    ]

    if st.button("Predict Churn"):

        response = requests.post(
            "https://customer-churn-prediction-ajzn.onrender.com/predict",
            json={"features": features}
        )

        if response.status_code == 200:

            result = response.json()

            prediction = result["prediction"]
            probability = result["churn_probability"]

            st.success(f"Prediction: {prediction}")

            st.write("Churn Probability")

            st.progress(probability)

            st.write(f"{probability:.2%}")

        else:
            st.error("API request failed")


# ---------------- BATCH PREDICTION ---------------- #

with tab2:

    st.subheader("Upload Dataset for Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Fix dataset columns automatically
        df = df.reindex(columns=expected_columns, fill_value=0)

        st.write("Preview of Uploaded Data")

        st.dataframe(df.head())

        if st.button("Run Predictions"):

            predictions = []
            probabilities = []

            with st.spinner("Running predictions..."):

                for _, row in df.iterrows():

                    features = row.tolist()

                    response = requests.post(
                        "http://127.0.0.1:8000/predict",
                        json={"features": features}
                    )

                    result = response.json()

                    predictions.append(result["prediction"])
                    probabilities.append(result["churn_probability"])

            df["churn_prediction"] = predictions
            df["churn_probability"] = probabilities

            st.success("Predictions completed!")

            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇ Download Predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )