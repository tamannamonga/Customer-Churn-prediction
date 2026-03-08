# Customer Churn Prediction System

## Live App : 
https://customer-churn-prediction-kvt7axbg8fdjchpqqvvglv.streamlit.app/

## Overview

This project predicts whether a telecom customer is likely to **churn (leave the service)** using machine learning. Customer churn prediction helps telecom companies identify customers who may cancel their subscription so they can take preventive actions such as offering discounts or improving services.

The project implements a complete **end-to-end machine learning pipeline**, including data preprocessing, model training, evaluation, API development, and an interactive user interface for predictions.

Users can either:

* Predict churn for a **single customer**
* Upload a **CSV or Excel dataset for batch predictions**

---

## Features

* Data cleaning and preprocessing
* Feature engineering and encoding
* Model comparison

  * Logistic Regression
  * Random Forest
  * Decision Tree
* Model evaluation using classification metrics
* Feature importance analysis
* FastAPI backend for prediction API
* Streamlit frontend for user interface
* Manual churn prediction
* Batch prediction using CSV/Excel file upload
* Downloadable prediction results

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Streamlit
* Joblib

---

## Dataset

The model was trained using the **Telco Customer Churn dataset**, which contains customer information such as tenure, monthly charges, contract type, and service usage.

The goal is to predict whether a customer will churn based on these attributes.

---

## Project Structure

```
customer-churn-prediction
│
├── api.py                # FastAPI backend for predictions
├── app.py                # Streamlit frontend application
├── churn_model.pkl       # Trained machine learning model
├── scaler.pkl            # Feature scaler used during training
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
```

---

## Machine Learning Workflow

### 1. Data Cleaning

* Converted `TotalCharges` to numeric values
* Removed missing values
* Removed unnecessary columns

### 2. Feature Encoding

* Converted categorical variables using one-hot encoding
* Binary features converted to numeric format

### 3. Feature Scaling

* Applied `StandardScaler` to normalize numerical features

### 4. Model Training

The following models were trained and compared:

* Logistic Regression
* Random Forest
* Decision Tree

### 5. Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

### 6. Model Selection

Logistic Regression performed best for this dataset and was selected as the final model.

---

## Application Features

### Manual Prediction

Users can enter customer information such as:

* Tenure
* Monthly Charges
* Contract Type
* Internet Service
* Payment Method

The system then predicts whether the customer is likely to churn and displays the churn probability.

---

### Batch Prediction

Users can upload a **CSV or Excel file** containing customer data.

The system:

1. Processes the uploaded dataset
2. Predicts churn for each row
3. Adds two new columns:

   * `churn_prediction`
   * `churn_probability`
4. Allows users to download the updated dataset

---

## Example Output

| tenure | MonthlyCharges | churn_prediction    | churn_probability |
| ------ | -------------- | ------------------- | ----------------- |
| 12     | 70             | Customer will churn | 0.63              |
| 48     | 45             | Customer will stay  | 0.12              |

---

## Author

**Tamanna Monga**
