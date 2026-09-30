# ISDA — Intelligent System for Data-Driven Aquaculture

A GIBC V2 hackathon submission by **Strelitzia Studios**.

A one-page web app: A farmer enters pond pH, temperature and turbidity, a RandomForest model recommends a fish species, and Gemini turns that into a short care report. No accounts, no login — the app is entirely stateless.

## What's in this folder

| File                | Purpose                                                |
| ------------------- | ------------------------------------------------------ |
| `fish.py`           | FastAPI backend — serves both the API and `index.html` |
| `index.html`        | The web app (Tailwind CSS, no build step)              |
| `fish_model.joblib` | Pre-trained RandomForest model                         |
| `requirements.txt`  | Runtime Python dependencies                            |

## Built with

FastAPI · Uvicorn · Pydantic · pandas · scikit-learn (RandomForestClassifier) · joblib · Google Gemini API (`google-genai`, `gemini-3.8-flash`) · Tailwind CSS (CDN) · vanilla JavaScript (no frontend framework/build step).

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn fish:app --host 0.0.0.0 --port 8000
```

_References_ (Data Sets)

Islam, M. M. (2023). Real-time dataset of pond water for fish farming using IoT devices. Data in Brief, 51, 109761. https://doi.org/10.1016/j.dib.2023.109761

Kashem, M. A. (2021). Real-time pond water dataset for fish farming [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/1669409
