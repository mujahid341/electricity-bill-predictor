from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load trained model and scaler
model = joblib.load("Linear_electricity_bill_model.pkl")
scaler = joblib.load("scaler.pkl")

# Input data schema without No_of_Family_Members
class BillData(BaseModel):
    Units_Consumed: int
    AC_Usage_Hours_per_Day: float
    Heater_Usage_Hours_per_Day: float

# POST route for prediction
@app.post("/predict")
def predictBill(data: BillData):
    # Convert input to NumPy array
    input_data = np.array([[
        data.Units_Consumed,
        data.AC_Usage_Hours_per_Day,
        data.Heater_Usage_Hours_per_Day
    ]])

    # Scale the input
    scaled_input = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(scaled_input)

    # Return the result
    return {"Predicted bill": round(prediction[0], 2)}
