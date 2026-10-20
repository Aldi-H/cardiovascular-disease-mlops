"""Upload a versioned preprocessing result to the private B2 bucket."""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config


REQUIRED_OUTPUTS = (
    "X_train.csv",
    "X_test.csv",
    "y_train.csv",
    "y_test.csv",
    "preprocessor.joblib",
    "preprocessing_metadata.json",
)
DEFAULT_PREFIX = "datasets/preprocessed"
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def upload_preprocessed_artifact(
    s3_client,
    bucket: str,
    input_dir: Path,
    run_id: str,
    commit_sha: str,
    prefix: str = DEFAULT_PREFIX,
) -> dict:
    """Upload required output files and atomically update the latest manifest."""
    if not SAFE_COMPONENT.fullmatch(run_id):
        raise ValueError("Run ID mengandung karakter yang tidak diizinkan.")
    if not SAFE_COMPONENT.fullmatch(commit_sha):
        raise ValueError("Commit SHA mengandung karakter yang tidak diizinkan.")

    input_dir = Path(input_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (input_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Output preprocessing tidak lengkap: {', '.join(missing)}"
        )

    clean_prefix = prefix.strip("/")
    version_prefix = f"{clean_prefix}/runs/{run_id}-{commit_sha[:12]}"
    file_keys = {}

    for filename in REQUIRED_OUTPUTS:
        object_key = f"{version_prefix}/{filename}"
        s3_client.upload_file(str(input_dir / filename), bucket, object_key)
        file_keys[filename] = object_key

    manifest = {
        "bucket": bucket,
        "prefix": version_prefix,
        "run_id": run_id,
        "commit_sha": commit_sha,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": file_keys,
    }
    manifest_key = f"{clean_prefix}/latest.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        ContentType="application/json",
    )
    manifest["manifest_key"] = manifest_key
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
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
        raise RuntimeError(f"Variabel akses B2 belum diisi: {missing_env}")

    s3_client = boto3.client(
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
    manifest = upload_preprocessed_artifact(
        s3_client=s3_client,
        bucket=os.environ["B2_BUCKET_NAME"],
        input_dir=args.input_dir,
        run_id=args.run_id,
        commit_sha=args.commit_sha,
        prefix=args.prefix,
    )

    print(f"Processed dataset uploaded to private B2: {manifest['prefix']}")
    print(f"Latest manifest key: {manifest['manifest_key']}")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as summary:
            summary.write("## Preprocessed dataset\n\n")
            summary.write(f"- B2 prefix: `{manifest['prefix']}`\n")
            summary.write(f"- Latest manifest: `{manifest['manifest_key']}`\n")


if __name__ == "__main__":
    main()
