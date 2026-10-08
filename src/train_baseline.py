import os
import time
from pathlib import Path
from dotenv import load_dotenv

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)


def load_processed_data(data_dir: Path):
    files = {
        "X_train": data_dir / "X_train.csv",
        "X_test": data_dir / "X_test.csv",
        "y_train": data_dir / "y_train.csv",
        "y_test": data_dir / "y_test.csv",
    }

    missing_files = [str(path) for path in files.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "File preprocessing belum lengkap: \n" + "\n".join(missing_files)
        )

    X_train = pd.read_csv(files["X_train"])
    X_test = pd.read_csv(files["X_test"])
    y_train_frame = pd.read_csv(files["y_train"])
    y_test_frame = pd.read_csv(files["y_test"])

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Kolom X_train dan X_test tidak sama.")

    if list(y_train_frame.columns) != ["cardio"]:
        raise ValueError("y_train.csv harus memiliki satu kolom bernama 'cardio'")

    if list(y_test_frame.columns) != ["cardio"]:
        raise ValueError("y_test.csv harus memiliki satu kolom bernaka 'cardio'")

    y_train = y_train_frame["cardio"]
    y_test = y_test_frame["cardio"]

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Jumlah baris feature dan target tidak cocok")

    return X_train, X_test, y_train, y_test


def evaluate_on_test(model, X_test, y_test) -> dict[str, float]:
    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_probability)),
        "test_pr_auc": float(average_precision_score(y_test, y_probability)),
        "test_log_loss": float(log_loss(y_test, y_probability, labels=[0, 1])),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dotenv_path = project_root / ".env"
    load_dotenv(dotenv_path, override=False)

    missing_env = [
        name
        for name in ENV_VARS
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]

    if missing_env:
        raise RuntimeError(f"Variable tracking belum diisi di .env: {missing_env}")

    data_dir = project_root / "data" / "clean"
    X_train, X_test, y_train, y_test = load_processed_data(data_dir)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("cardiovascular-disease-baseline")
    mlflow.sklearn.autolog(log_models=True)

    models = {
        "logistic-regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=f"baseline-{name}") as run:
            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time_seconds = time.perf_counter() - start_time

            metrics = evaluate_on_test(model, X_test, y_test)
            metrics["fit_time_seconds"] = float(fit_time_seconds)

            mlflow.log_param("evaluation_threshold", 0.5)
            mlflow.log_metrics(metrics)

            print(f"\nModel: {name}")
            print(f"Run ID: {run.info.run_id}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
