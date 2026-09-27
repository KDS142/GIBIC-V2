"""
ISDA — Intelligent System for Data-Driven Aquaculture
------------------------------------------------------
Predicts a suitable fish species for a pond from sensor readings (pH,
temperature, turbidity) using a pre-trained RandomForest model, then asks
Gemini to turn that prediction into a short, farmer-friendly report.

"""

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google import genai
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Setup & configuration
# --------------------------------------------------------------------------

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("fish-api")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fish_model.joblib"
FRONTEND_PATH = BASE_DIR / "index.html"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]
ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "true").lower() == "true"
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
GEMINI_TIMEOUT_SECONDS = float(os.environ.get("GEMINI_TIMEOUT_SECONDS", "20"))

if not GEMINI_API_KEY:
    logger.warning(
        "GEMINI_API_KEY is not set. Predictions will still work, but every "
        "response will use the fallback advisory message instead of an "
        "AI-generated one. Add it to a .env file (see .env.example)."
    )

if ALLOWED_ORIGINS == [origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
    ]:
    logger.warning(
        "ALLOWED_ORIGINS is not set, defaulting to '*'. This is fine for "
        "local development but should be restricted to your real frontend "
        "domain(s) before this is deployed anywhere public."
    )

# Security: The Gemini client and API key are strictly isolated on the backend. Neither is ever exposed to the frontend or leaked in API responses.
genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

try:
    loaded_model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH.name)
except Exception as exc:  # noqa: BLE001 (No quality assurance)- fail fast and loud at startup
    raise RuntimeError(f"Failed to load model file '{MODEL_PATH.name}': {exc}") from exc

FEATURE_NAMES = ["ph", "temperature", "turbidity"]

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(
    title="ISDA — Intelligent System for Data-Driven Aquaculture",
    description="Receives pond sensor readings, predicts a suitable fish "
    "species, and returns an AI-generated advisory report.",
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# --------------------------------------------------------------------------
# In-memory, per-process rate limiter to protect the ML model and Gemini API quota from abuse.
# Note: State is not shared across multiple worker processes. It is designed for single-instance use.
# Multi-worker environments require an external rate limiter (like Render's Redis).
# --------------------------------------------------------------------------

_request_log: Dict[str, Deque[float]] = defaultdict(deque)
_cleanup_counter = 0
# Periodically clears expired IP entries from the rate limiter. This prevents unbounded memory growth 
# (memory leaks) by ensuring the request log doesn't retain inactive client histories indefinitely.
_CLEANUP_EVERY = 500

async def cleanup_stale_ips():
    now = time.monotonic()
    stale_ips = [
        ip for ip, hist in list(_request_log.items())
        if not hist or now - hist[-1] > RATE_LIMIT_WINDOW_SECONDS
    ]
    for ip in stale_ips:
        _request_log.pop(ip, None)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    global _cleanup_counter
    if request.url.path == "/api/predict-and-advise":
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        history = _request_log[client_ip]
        while history and now - history[0] > RATE_LIMIT_WINDOW_SECONDS:
            history.popleft()
        if len(history) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests. Please wait a moment and try again."},
            )
        history.append(now)

        _cleanup_counter += 1
        if _cleanup_counter >= _CLEANUP_EVERY:
            _cleanup_counter = 0
            asyncio.create_task(cleanup_stale_ips())
    return await call_next(request)

# --------------------------------------------------------------------------
# Schemas / Input validation and output formatting
# --------------------------------------------------------------------------

class FarmDataRequest(BaseModel):
    ph: float = Field(..., ge=0.0, le=14.0, description="Pond pH level (0-14)")
    temperature: float = Field(..., ge=0.0, le=50.0, description="Water temperature in Celsius")
    turbidity: float = Field(..., ge=0.0, le=3000.0, description="Water turbidity in NTU")
    pond_size_sqm: float = Field(
        500.0, gt=0.0, le=1_000_000.0, description="Pond surface area in square meters"
    )
    pond_depth_m: float = Field(
        1.2,
        gt=0.0,
        le=50.0,
        description="Average pond depth in meters, used only to estimate water volume for the advisory",
    )

class FarmDataResponse(BaseModel):
    recommended_fish: str
    confidence_percentage: float
    ai_advisory: str
    ai_advisory_available: bool

# --------------------------------------------------------------------------
# Error handling — never leak stack traces / internal details to clients.
# --------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        field = ".".join(str(loc) for loc in err["loc"] if loc != "body")
        errors.append({"field": field or "request", "message": err["msg"]})
    logger.info("Rejected invalid request to %s: %s", request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"detail": "Some of those readings look invalid.", "errors": errors},
    )

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Lightweight liveness/readiness check — safe to expose publicly."""
    return {
        "status": "ok",
        "model_loaded": loaded_model is not None,
        "ai_advisory_enabled": genai_client is not None,
    }

@app.get("/")
def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(FRONTEND_PATH)
    raise HTTPException(status_code=404, detail="Frontend file not found next to the server.")

@app.post("/api/predict-and-advise", response_model=FarmDataResponse)
async def predict_and_advise(data: FarmDataRequest):
    # --- 1. ML prediction: A fast, CPU-bound step that must succeed before triggering the AI ---
    try:
        input_df = pd.DataFrame([[data.ph, data.temperature, data.turbidity]], columns=FEATURE_NAMES)
        prediction = str(loaded_model.predict(input_df)[0])
        probabilities = loaded_model.predict_proba(input_df)[0]
        confidence = float(max(probabilities) * 100)
    except Exception:
        # Security: Log the full error internally to prevent leaking sensitive exception details to the client.
        logger.exception("Model prediction failed for input=%s", data.model_dump())
        raise HTTPException(
            status_code=500,
            detail="We couldn't generate a recommendation from those readings. Please try again.",
        )

    fallback_advisory = (
        f"Recommended species: {prediction} (confidence {confidence:.1f}%). "
        "AI-generated tips are temporarily unavailable, but this recommendation "
        "is based directly on your pond's pH, temperature and turbidity readings."
    )

    # --- 2. AI advisory: Fault-tolerant external API call. Network timeouts or Gemini quota limits will fail safely, 
    # ensuring the farmer still receives their ML prediction instead of a generic 500 server error.
    if genai_client is None:
        return FarmDataResponse(
            recommended_fish=prediction,
            confidence_percentage=round(confidence, 2),
            ai_advisory=fallback_advisory,
            ai_advisory_available=False,
        )

    volume_m3 = data.pond_size_sqm * data.pond_depth_m

    prompt = f"""You are an aquaculture advisor providing a concise report for a farmer's mobile app.

Pond Sensor Readings:
- pH: {data.ph}
- Temperature: {data.temperature} degrees C
- Turbidity: {data.turbidity} NTU
- Pond Surface Area: {data.pond_size_sqm} sqm
- Average Depth: {data.pond_depth_m} m
- Approximate Water Volume: {volume_m3:,.0f} cubic meters (~{volume_m3 * 1000:,.0f} liters)

Machine Learning Prediction:
- Recommended Fish: {prediction}
- Confidence: {confidence:.1f}%

Provide an easy-to-read, encouraging assessment covering:
1. Water Quality Status (safe vs caution)
2. Expected Harvest/Stocking Yield for this pond size
3. Two key daily maintenance tips
"""
    try:
        ai_response = await asyncio.wait_for(
            genai_client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            ),
            timeout=GEMINI_TIMEOUT_SECONDS,
        )
        advisory_text = (getattr(ai_response, "text", "") or "").strip() or fallback_advisory
        return FarmDataResponse(
            recommended_fish=prediction,
            confidence_percentage=round(confidence, 2),
            ai_advisory=advisory_text,
            ai_advisory_available=True,
        )
    except Exception:
        # Catches and suppresses timeouts, quota errors, and any Gemini/network failure.
        logger.exception("Gemini advisory call failed or timed out")
        return FarmDataResponse(
            recommended_fish=prediction,
            confidence_percentage=round(confidence, 2),
            ai_advisory=fallback_advisory,
            ai_advisory_available=False,
        )