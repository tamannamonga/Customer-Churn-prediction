from fastapi import FastAPI
from pydantic import BaseModel
import joblib
app = FastAPI() #creates api application.
# loading the model and scaler.
model=joblib.load("churn_model.pkl")
scaler=joblib.load("scaler.pkl")
class CustomerData(BaseModel):  # for defining input format.
    features : list
@app.get("/") #endpoint
def home():
    return {"message":"customer churn prediction api is running"}
@app.post("/predict") #post endpoint
def predict(data : CustomerData):
    input_data = data.features
    input_data = scaler.transform([input_data])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    if prediction[0] == 1:
        result = "customer will churn"
    else:
        result = "customer will stay"
    return {
        "prediction": result,
        "churn_probability": float(probability)
    }
