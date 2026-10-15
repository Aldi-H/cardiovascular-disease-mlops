import pytest
from pydantic import ValidationError

from monitoring.traffic_simulator import make_payload
from serving.app import PredictionRequest


@pytest.mark.parametrize("scenario", ["normal", "drift"])
def test_normal_and_drift_payloads_are_valid(scenario):
    payload = make_payload(scenario, index=0, seed=42)

    request = PredictionRequest(**payload)

    assert request.ap_hi >= request.ap_lo


def test_invalid_scenario_fails_request_validation():
    payload = make_payload("invalid", index=0, seed=42)

    with pytest.raises(ValidationError, match="ap_hi"):
        PredictionRequest(**payload)
