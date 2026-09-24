import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai

app = FastAPI(
    title="Fish Farm ML & AI Advisory API",
    description="Endpoint to receive farm parameters, predict suitable species, and generate AI insights."
)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

model_filename = "fish_model.joblib"
try:
    loaded_model = joblib.load(model_filename)
except Exception as e:
    raise RuntimeError(f"Failed to load model file '{model_filename}': {e}")

class FarmDataRequest(BaseModel):
    ph: float = Field(..., ge=0.0, le=14.0, description="Pond pH level (0-14)")
    temperature: float = Field(..., ge=0.0, le=50.0, description="Water temperature in Celsius")
    turbidity: float = Field(..., ge=0.0, description="Water turbidity in NTU")
    pond_size_sqm: float = Field(500.0, gt=0.0, description="Pond surface area in square meters")

class FarmDataResponse(BaseModel):
    recommended_fish: str
    confidence_percentage: float
    ai_advisory: str

@app.post("/api/predict-and-advise", response_model=FarmDataResponse)
def predict_and_advise(data: FarmDataRequest):
    try:
        feature_names = ['ph', 'temperature', 'turbidity']
        input_df = pd.DataFrame([[data.ph, data.temperature, data.turbidity]], columns=feature_names)
        prediction = loaded_model.predict(input_df)[0]
        probabilities = loaded_model.predict_proba(input_df)[0]
        confidence = float(max(probabilities) * 100)
        prompt = f"""
        You are an aquaculture advisor providing a concise report for a farmer's mobile app.

        Pond Sensor Readings:
        - pH: {data.ph}
        - Temperature: {data.temperature} °C
        - Turbidity: {data.turbidity} NTU
        - Pond Area: {data.pond_size_sqm} sqm

        Machine Learning Prediction:
        - Recommended Fish: {prediction}
        - Confidence: {confidence:.1f}%

        Provide an easy-to-read, encouraging assessment covering:
        1. Water Quality Status (safe vs caution)
        2. Expected Harvest/Stocking Yield for this pond size
        3. 2 key daily maintenance tips
        """

        ai_response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return FarmDataResponse(
            recommended_fish=prediction,
            confidence_percentage=round(confidence, 2),
            ai_advisory=ai_response.text
        )

    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))