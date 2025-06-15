from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Create FastAPI app
app = FastAPI()

# Allow all frontend origins for testing (adjust in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Student Score Predictor API is running"}

@app.get("/predict")
def predict(hours: float = Query(..., description="Number of study hours")):
    input_data = np.array([[hours]])  # 2D array for sklearn
    prediction = model.predict(input_data)
    predicted_score = round(float(prediction[0]), 2)
    return {"predicted_score": predicted_score}
