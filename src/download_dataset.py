"""Download the raw cardiovascular dataset from a private Backblaze B2 bucket."""

import argparse
import os
import tempfile
from pathlib import Path

import boto3
import pandas as pd
from botocore.config import Config
from dotenv import load_dotenv


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


def main(argv: list[str] | None = None) -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env", override=False)

    parser = argparse.ArgumentParser(
        description="Download the raw cardiovascular CSV from Backblaze B2."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "raw" / "cardiovascular-disease.csv",
        help="Destination for the downloaded CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args(argv)

    required_env = (
        "MLFLOW_S3_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "B2_BUCKET_NAME",
    )
    missing_env = [
        name
        for name in required_env
        if not os.getenv(name) or os.getenv(name).startswith("<")
    ]
    if missing_env:
        raise RuntimeError(f"Variabel B2 belum diisi: {missing_env}")

    bucket = os.environ["B2_BUCKET_NAME"]
    object_key = os.getenv(
        "B2_DATA_OBJECT_KEY",
        "datasets/cardiovascular-disease.csv",
    ).lstrip("/")

    if args.output.exists() and not args.force:
        parser.error(
            f"File sudah ada: {args.output}. Gunakan --output ke lokasi baru "
            "atau --force untuk menggantinya."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            prefix=".cardio-download-",
            suffix=".csv",
            dir=args.output.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name=os.environ["AWS_DEFAULT_REGION"],
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        s3.download_file(bucket, object_key, str(temp_path))

        data = pd.read_csv(temp_path, sep=";")
        if data.shape[1] == 1 and "," in str(data.columns[0]):
            data = pd.read_csv(temp_path)

        missing_columns = REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            raise ValueError(
                f"CSV tidak memiliki kolom wajib: {sorted(missing_columns)}"
            )
        if data.empty:
            raise ValueError("CSV yang diunduh kosong.")

        os.replace(temp_path, args.output)
        temp_path = None

        print(f"Download berhasil: s3://{bucket}/{object_key}")
        print(f"Ukuran data: {data.shape[0]:,} baris × {data.shape[1]} kolom")
        print(f"Disimpan ke: {args.output}")
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
