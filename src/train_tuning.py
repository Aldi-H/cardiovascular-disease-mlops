import json
import os
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    BINARY_FEATURES,
    CANDIDATE_BOUNDS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    clean_dataframe,
)

ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
DECISION_THRESHOLD = 0.5


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)

    classes = pipeline.named_steps["model"].classes_
    positive_index = int(np.flatnonzero(classes == 1)[0])
    y_score = probabilities[:, positive_index]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_score)),
        "test_pr_auc": float(average_precision_score(y_test, y_score)),
        "test_log_loss": float(log_loss(y_test, y_score, labels=[0, 1])),
        "test_false_positives": float(fp),
        "test_false_negatives": float(fn),
        "test_true_positives": float(tp),
        "test_true_negatives": float(tn),
    }
    return metrics, y_pred, y_score


def log_evaluation_artifacts(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    cv_results: pd.DataFrame,
    metrics: dict[str, float],
    model_name: str,
) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        confusion_path = temp_path / "confusion_matrix.png"
        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=[0, 1],
            cmap="Blues",
            values_format="d",
        )
        display.ax_.set_title(f"Confusion Matrix - {model_name}")
        display.figure_.tight_layout()
        display.figure_.savefig(confusion_path, dpi=150, bbox_inches="tight")
        plt.close(display.figure_)
        mlflow.log_artifact(str(confusion_path), artifact_path="evaluation")

        curves_path = temp_path / "roc_pr_curves.png"
        fpr, tpr, _ = roc_curve(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, y_score)

        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr)
        axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
        axes[0].set(
            title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate"
        )

        axes[1].plot(recall, precision)
        axes[1].set(
            title="Precision-Recall Curve",
            xlabel="Recall",
            ylabel="Precision",
        )

        figure.suptitle(f"Evaluation Curves - {model_name}")
        figure.tight_layout()
        figure.savefig(curves_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        mlflow.log_artifact(str(curves_path), artifact_path="evaluation")

        cv_results_path = temp_path / "cv_results.csv"
        cv_results.to_csv(cv_results_path, index=False)
        mlflow.log_artifact(str(cv_results_path), artifact_path="evaluation")

        metric_info = {
            "positive_class": 1,
            "decision_threshold": DECISION_THRESHOLD,
            "selection_metric": "mean 5-fold CV recall",
            "metrics": metrics,
            "definitions": {
                "recall": "TP / (TP + FN); primary metric for reducing false negatives.",
                "f1": "Harmonic mean of precision and recall.",
                "roc_auc": "Area under the ROC Curve.",
                "pr_auc": "Average precision over the precision-recall curve.",
                "log_loss": "Logarithmic loss from predicted probabilites.",
            },
            "candidate_bounds": {
                key: list(value) for key, value in CANDIDATE_BOUNDS.items()
            },
        }

        mlflow.log_dict(metric_info, "evaluation/metric_info.json")


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    missing_env = [
        name
        for name in ENV_VARS
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]

    if missing_env:
        raise RuntimeError(f"Variable tracking belum diisi di .env: {missing_env}")

    raw_path = PROJECT_ROOT / "data" / "raw" / "cardiovascular-disease.csv"
    if not raw_path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {raw_path}")

    raw = pd.read_csv(raw_path, sep=";")
    if raw.shape[1] == 1 and "," in str(raw.columns[0]):
        raw = pd.read_csv(raw_path)

    cleaned = clean_dataframe(raw)
    X = cleaned.drop(columns=["cardio"])
    y = cleaned["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scoring = {"recall": "recall", "f1": "f1"}

    model_searches = {
        "logistic-regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            [
                {
                    "model__n_estimators": [50],
                    "model__max_depth": [8],
                    "model__min_samples_leaf": [2],
                    "model__class_weight": [None],
                },
                {
                    "model__n_estimators": [100],
                    "model__max_depth": [12],
                    "model__min_samples_leaf": [2],
                    "model__class_weight": ["balanced_subsample"],
                },
            ],
        ),
    }

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("cardiovascular-disease-tuning-b2")

    for model_name, (estimator, param_grid) in model_searches.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("model", estimator)]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            refit="recall",
            cv=cv,
            n_jobs=1,
            verbose=1,
            return_train_score=False,
            error_score="raise",
        )

        run_name = f"tuning-{model_name}"
        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.perf_counter()
            search.fit(X_train, y_train)
            search_time = time.perf_counter() - start_time

            best_index = search.best_index_
            cv_results = pd.DataFrame(search.cv_results_)
            best_pipeline = search.best_estimator_

            test_metrics, y_pred, y_score = evaluate_model(
                best_pipeline, X_test, y_test
            )

            metrics = {
                **test_metrics,
                "cv_recall_mean": float(
                    search.cv_results_["mean_test_recall"][best_index]
                ),
                "cv_recall_std": float(
                    search.cv_results_["std_test_recall"][best_index]
                ),
                "cv_f1_mean": float(search.cv_results_["mean_test_f1"][best_index]),
                "cv_f1_std": float(search.cv_results_["std_test_f1"][best_index]),
                "search_fit_time_seconds": float(search_time),
            }

            manual_params = {
                "model_name": model_name,
                "cv_type": "StratifiedKFold",
                "cv_splits": 5,
                "cv_selection_metric": "recall",
                "decision_threshold": DECISION_THRESHOLD,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            }

            manual_params.update(
                {
                    f"best_{key}": str(value)
                    for key, value in search.best_params_.items()
                }
            )

            mlflow.log_params(manual_params)
            mlflow.log_metrics(metrics)

            input_example = pd.DataFrame(
                {
                    "age_years": [50.0],
                    "height": [165],
                    "weight": [72.0],
                    "ap_hi": [120],
                    "ap_lo": [80],
                    "cholesterol": [1],
                    "gluc": [1],
                    "smoke": [0],
                    "alco": [0],
                    "active": [1],
                    "bmi": [26.45],
                    "gender": [1],
                },
                columns=X_train.columns,
            )
            input_example = input_example.astype("float64")
            signature = mlflow.models.infer_signature(
                input_example,
                best_pipeline.predict(input_example),
            )

            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            log_evaluation_artifacts(
                y_test=y_test,
                y_pred=y_pred,
                y_score=y_score,
                cv_results=cv_results,
                metrics=metrics,
                model_name=model_name,
            )

            print(f"\nModel: {model_name}")
            print(f"Run ID: {run.info.run_id}")
            print(f"Best Params: {search.best_params_}")
            print(f"CV Recall: {metrics['cv_recall_mean']:.4f}")
            print(f"CV F1: {metrics['cv_f1_mean']:.4f}")
            print(f"Test Recall: {metrics['test_recall']:.4f}")
            print(f"Test F1: {metrics['test_f1']:.4f}")


if __name__ == "__main__":
    main()
