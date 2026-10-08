import os
import tempfile
from pathlib import Path

import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_DEFAULT_REGION",
    "B2_BUCKET_NAME",
)
EXPERIMENT_NAME = "cardiovascular-disease-tuning-b2"
ARTIFACT_PREFIX = "mlflow-artifacts"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    missing_env = [
        name
        for name in REQUIRED_ENV_VARS
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]
    if missing_env:
        raise RuntimeError(f"Variabel belum diisi di .env: {missing_env}")

    bucket_name = os.environ["B2_BUCKET_NAME"]
    artifact_location = f"s3://{bucket_name}/{ARTIFACT_PREFIX}"

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_location,
        )
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment.artifact_location.rstrip("/") != artifact_location.rstrip("/"):
        raise RuntimeError(
            f"Experiment {EXPERIMENT_NAME!r} sudah ada dengan artifact location "
            f"{experiment.artifact_location!r}, bukan {artifact_location!r}. "
            "Gunakan nama experiment baru; artifact location experiment yang sudah "
            "ada tidak diganti oleh script ini."
        )

    mlflow.set_experiment(EXPERIMENT_NAME)

    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "b2-smoke-test.txt"
        test_file.write_text(
            "Backblaze B2 artifact upload test\n",
            encoding="utf-8",
        )

        with mlflow.start_run(run_name="b2-artifact-smoke-test") as run:
            mlflow.log_param("purpose", "verify_b2_artifact_upload")
            mlflow.log_metric("upload_test", 1.0)
            mlflow.log_artifact(str(test_file), artifact_path="smoke-test")
            run_id = run.info.run_id

    completed_run = mlflow.get_run(run_id)
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Artifact location: {completed_run.info.artifact_uri}")
    print(f"Run ID: {run_id}")
    print("Smoke test berhasil dikirim.")


if __name__ == "__main__":
    main()
