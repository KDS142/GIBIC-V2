# ISDA — Intelligent System for Data-Driven Aquaculture

A GIBC V2 hackathon submission by **Strelitzia Studios**.

A one-page web app: A farmer enters pond pH, temperature and turbidity, a RandomForest model recommends a fish species, and Gemini turns that into a short care report. No accounts, no login — the app is entirely stateless.

## What's in this folder

| File | Purpose |
|---|---|
| `fish.py` | FastAPI backend — serves both the API and `index.html` |
| `index.html` | The web app (Tailwind CSS, no build step) |
| `fish_model.joblib` | Pre-trained RandomForest model |
| `requirements.txt` | Runtime Python dependencies |

## Built with
FastAPI · Uvicorn · Pydantic · pandas · scikit-learn (RandomForestClassifier) · joblib · Google Gemini API (`google-genai`, `gemini-2.5-flash`) · Tailwind CSS (CDN) · vanilla JavaScript (no frontend framework/build step).

## Run it
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn fish:app --host 0.0.0.0 --port 8000
```