import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CANDIDATE_BOUNDS = {
    "ap_hi": (70, 250),
    "ap_lo": (40, 150),
    "height": (120, 220),
    "weight": (30, 250),
}

REQUIRED_COLUMNS = {
    "id",
    "age",
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
    "cardio",
}


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)

    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing_columns)}")

    missing_values = raw[list(REQUIRED_COLUMNS)].isna().sum()
    columns_with_missing = missing_values[missing_values > 0]

    if not columns_with_missing.empty:
        raise ValueError(
            "Nilai kosong ditemukan; tentukan kebijakan penanganannya terlebih dahulu: "
            f"{columns_with_missing.to_dict()}"
        )

    valid_mask = raw["ap_hi"].gt(0) & raw["ap_lo"].gt(0) & raw["ap_hi"].ge(raw["ap_lo"])

    for column, (lower, upper) in CANDIDATE_BOUNDS.items():
        valid_mask &= raw[column].between(lower, upper, inclusive="both")

    cleaned = raw.loc[valid_mask].copy()
    cleaned["age_years"] = cleaned["age"] / 365.25
    cleaned["bmi"] = cleaned["weight"] / (cleaned["height"] / 100) ** 2

    return cleaned.drop(columns=["id", "age"])


NUMERIC_FEATURES = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]

CATEGORICAL_FEATURES = ["gender", "cholesterol", "gluc"]

BINARY_FEATURES = ["smoke", "alco", "active"]


def split_and_transform(
    cleaned: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    X = cleaned.drop(columns=["cardio"])
    y = cleaned["cardio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    ct = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )

    X_train_processed = ct.fit_transform(X_train)
    X_test_processed = ct.transform(X_test)

    return (X_train_processed, X_test_processed, y_train, y_test, ct)


def matrix_to_dataframe(matrix, feature_names: list[str]) -> pd.DataFrame:
    """Convert dense or sparse transformed features to a named DataFrame."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return pd.DataFrame(matrix, columns=feature_names)


def main(argv: list[str] | None = None) -> None:
    """Run preprocessing from the command line and persist the outputs."""
    project_root = Path(__file__).resolve().parents[1]
    default_input = project_root / "data" / "raw" / "cardiovascular-disease.csv"
    default_output = project_root / "data" / "clean"

    parser = argparse.ArgumentParser(
        description="Clean and split the cardiovascular disease dataset."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Input CSV path (default: {default_input})",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Directory for processed data (default: {default_output})",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of cleaned rows assigned to the test set (default: 0.2).",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for the stratified split (default: 42).",
    )

    args = parser.parse_args(argv)

    if not 0 < args.test_size < 1:
        parser.error("--test-size harus lebih besar dari 0 dan lebih kecil dari 1.")

    if not args.input.is_file():
        parser.error(f"File input tidak ditemukan: {args.input}")

    raw = pd.read_csv(args.input, sep=";")

    if raw.shape[1] == 1 and "," in str(raw.columns[0]):
        raw = pd.read_csv(args.input)

    cleaned = clean_dataframe(raw)

    X_train, X_test, y_train, y_test, preprocessor = split_and_transform(
        cleaned,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = preprocessor.get_feature_names_out().tolist()

    matrix_to_dataframe(X_train, feature_names).to_csv(
        args.output_dir / "X_train.csv", index=False
    )

    matrix_to_dataframe(X_test, feature_names).to_csv(
        args.output_dir / "X_test.csv", index=False
    )

    y_train.reset_index(drop=True).to_frame(name="cardio").to_csv(
        args.output_dir / "y_train.csv", index=False
    )

    y_test.reset_index(drop=True).to_frame(name="cardio").to_csv(
        args.output_dir / "y_test.csv", index=False
    )

    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")

    metadata = {
        "input_file": args.input.name,
        "raw_rows": int(len(raw)),
        "cleaned_rows": int(len(cleaned)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "feature_names": feature_names,
        "candidate_bounds": {
            column: list(bounds) for column, bounds in CANDIDATE_BOUNDS.items()
        },
    }

    with (args.output_dir / "preprocessing_metadata.json").open(
        "w", encoding="utf-8"
    ) as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"Data mentah: {raw.shape}")
    print(f"Data setelah cleaning: {cleaned.shape}")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    print(f"Hasil disimpan di: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
