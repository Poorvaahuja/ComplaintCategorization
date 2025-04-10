from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load the full pipeline model (includes vectorizer + classifier)
model = joblib.load("model\maintenance_classifier.pkl")

# FastAPI app instance
app = FastAPI()

# Request schema
class Complaint(BaseModel):
    complaint: str

@app.get("/")
def home():
    return {"message": "PG Complaint Categorization API is running."}

@app.post("/predict")
def predict_category(data: Complaint):
    complaint_text = data.complaint
    prediction = model.predict([complaint_text])[0]
    return {
        "complaint": complaint_text,
        "predicted_category": prediction
    }
