import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import serving.app as serving_app
from serving.app import MODEL_FEATURES, PredictionRequest, build_model_features


VALID_PAYLOAD = {
    "age_days": 18_393,
    "gender": 2,
    "height": 168,
    "weight": 62.0,
    "ap_hi": 110,
    "ap_lo": 80,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}


class DummyClassifier:
    classes_ = np.array([0, 1])


class DummyPipeline:
    named_steps = {"model": DummyClassifier()}

    def predict_proba(self, features):
        assert list(features.columns) == MODEL_FEATURES
        return np.array([[0.25, 0.75]])


@pytest.fixture
def api_client(monkeypatch):
    test_env = {
        "MLFLOW_TRACKING_URI": "https://example.invalid/mlflow",
        "MLFLOW_TRACKING_USERNAME": "test-user",
        "MLFLOW_TRACKING_PASSWORD": "test-token",
        "MLFLOW_S3_ENDPOINT_URL": "https://s3.example.invalid",
        "AWS_ACCESS_KEY_ID": "test-key-id",
        "AWS_SECRET_ACCESS_KEY": "test-secret-key",
        "AWS_DEFAULT_REGION": "test-region",
        "MODEL_URI": "runs:/test-run/model",
    }
    for name, value in test_env.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(
        serving_app.mlflow.sklearn,
        "load_model",
        lambda model_uri: DummyPipeline(),
    )

    with TestClient(serving_app.app) as client:
        yield client


def test_build_model_features_converts_age_and_adds_bmi():
    payload = PredictionRequest(**VALID_PAYLOAD)

    features = build_model_features(payload)

    assert list(features.columns) == MODEL_FEATURES
    assert features.loc[0, "age_years"] == pytest.approx(18_393 / 365.25)
    assert features.loc[0, "bmi"] == pytest.approx(62 / (1.68**2))


def test_prediction_request_rejects_reversed_blood_pressure():
    invalid_payload = {**VALID_PAYLOAD, "ap_hi": 80, "ap_lo": 100}

    with pytest.raises(ValidationError, match="ap_hi"):
        PredictionRequest(**invalid_payload)


def test_prediction_request_rejects_target_as_input():
    with pytest.raises(ValidationError):
        PredictionRequest(**VALID_PAYLOAD, cardio=1)


def test_health_predict_and_metrics_endpoints(api_client):
    health = api_client.get("/health")
    prediction = api_client.post("/predict", json=VALID_PAYLOAD)
    metrics = api_client.get("/metrics")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is True

    assert prediction.status_code == 200
    assert prediction.json()["cardio_prediction"] == 1
    assert prediction.json()["probability_cardio"] == pytest.approx(0.75)

    assert metrics.status_code == 200
    assert "http_requests_total" in metrics.text
    assert "predictions_total" in metrics.text
