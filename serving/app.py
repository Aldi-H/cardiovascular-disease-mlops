import logging
import os
import time

from dotenv import load_dotenv
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import Response

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FEATURES = [
    "gender",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "age_years",
    "bmi",
]

DECISION_THRESHOLD = 0.5

logger = logging.getLogger("cardiovascular-serving")

HTTP_REQUESTS = Counter(
    "http_requests", "Jumlah HTTP request", ["method", "path", "status_code"]
)
HTTP_ERRORS = Counter(
    "http_errors", "Jumlah HTTP response error", ["path", "status_code"]
)
PREDICTION_ERRORS = Counter("predictions_errors", "Jumlah kegagalan prediksi")
PREDICTIONS = Counter("predictions", "Jumlah prediksi per class", ["cardio_class"])
PREDICTION_CONFIDENCE = Histogram(
    "prediction_confidence", "Probabilitas prediksi kelas cardio=1"
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durasi HTTP request dalam detik",
    ["method", "path"],
)
MODEL_LOAD_SECONDS = Gauge(
    "model_load_duration_seconds", "Durasi pemuatan model dalam detik"
)
IN_FLIGHT_REQUESTS = Gauge(
    "http_requests_in_flight", "Jumlah HTTP request yang sedang diproses"
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age_days: int = Field(ge=10_798, le=23_713)

    gender: Literal[1, 2]
    height: int = Field(ge=120, le=220)
    weight: float = Field(ge=30, le=250)
    ap_hi: int = Field(ge=70, le=250)
    ap_lo: int = Field(ge=40, le=150)
    cholesterol: Literal[1, 2, 3]
    gluc: Literal[1, 2, 3]
    smoke: Literal[0, 1]
    alco: Literal[0, 1]
    active: Literal[0, 1]

    @model_validator(mode="after")
    def validate_blood_pressure_order(self):
        if self.ap_hi < self.ap_lo:
            raise ValueError("ap_hi harus lebih besar atau sama dengan ap_lo")
        return self


def build_model_features(payload: PredictionRequest) -> pd.DataFrame:
    bmi = payload.weight / (payload.height / 100) ** 2

    row = {
        "gender": payload.gender,
        "height": payload.height,
        "weight": payload.weight,
        "ap_hi": payload.ap_hi,
        "ap_lo": payload.ap_lo,
        "cholesterol": payload.cholesterol,
        "gluc": payload.gluc,
        "smoke": payload.smoke,
        "alco": payload.alco,
        "active": payload.active,
        "age_years": payload.age_days / 365.25,
        "bmi": bmi,
    }

    return pd.DataFrame([row], columns=MODEL_FEATURES)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    required_env = (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "MODEL_URI",
    )

    missing_env = [
        name
        for name in required_env
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]

    if missing_env:
        raise RuntimeError(f"Variabel serving belum diisi: {missing_env}")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    start_time = time.perf_counter()
    app.state.model = mlflow.sklearn.load_model(os.environ["MODEL_URI"])
    MODEL_LOAD_SECONDS.set(time.perf_counter() - start_time)

    logger.info("Model dimuat.")
    yield


app = FastAPI(
    title="Cardiovascular Disease Screening API",
    description="Prototype screening.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def instrument_http_requests(request: Request, call_next):
    path = request.url.path
    status_code = "500"
    start_time = time.perf_counter()
    IN_FLIGHT_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        if response.status_code >= 400:
            HTTP_ERRORS.labels(path=path, status_code=status_code).inc()

        return response
    except Exception:
        HTTP_ERRORS.labels(path=path, status_code=status_code).inc()
        logger.exception("HTTP request failed.")
        raise
    finally:
        HTTP_REQUESTS.labels(
            method=request.method,
            path=path,
            status_code=status_code,
        ).inc()
        REQUEST_LATENCY.labels(method=request.method, path=path).observe(
            time.perf_counter() - start_time
        )
        IN_FLIGHT_REQUESTS.dec()


@app.get("/health")
def health(request: Request):
    model_loaded = getattr(request.app.state, "model", None) is not None
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not ready")

    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        features = build_model_features(payload)
        probabilities = model.predict_proba(features)[0]

        classifier = model.named_steps.get("model", model)
        classes = list(classifier.classes_)
        positive_index = classes.index(1)
        probability_cardio = float(probabilities[positive_index])
        prediction = int(probability_cardio >= DECISION_THRESHOLD)

        PREDICTIONS.labels(cardio_class=str(prediction)).inc()
        PREDICTION_CONFIDENCE.observe(probability_cardio)

        return {
            "cardio_prediction": prediction,
            "probability_cardio": probability_cardio,
            "decision_threshold": DECISION_THRESHOLD,
            "disclaimer": "hasil bukan diagnosis medis",
        }
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.exception("Inferensi gagal.")
        raise HTTPException(
            status_code=500, detail="Terjadi kesalahan saat melakukan prediksi."
        ) from exc


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
