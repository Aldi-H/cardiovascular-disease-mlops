import json

import pytest

from src.upload_preprocessed_to_b2 import (
    REQUIRED_OUTPUTS,
    upload_preprocessed_artifact,
)


class FakeS3Client:
    def __init__(self):
        self.uploaded = []
        self.objects = []

    def upload_file(self, filename, bucket, key):
        self.uploaded.append((filename, bucket, key))

    def put_object(self, **kwargs):
        self.objects.append(kwargs)


def test_uploads_preprocessed_files_and_latest_manifest(tmp_path):
    for filename in REQUIRED_OUTPUTS:
        (tmp_path / filename).write_text("test output", encoding="utf-8")
    client = FakeS3Client()

    manifest = upload_preprocessed_artifact(
        s3_client=client,
        bucket="private-bucket",
        input_dir=tmp_path,
        run_id="123456789",
        commit_sha="abcdef0123456789",
    )

    expected_prefix = "datasets/preprocessed/runs/123456789-abcdef012345"
    assert len(client.uploaded) == len(REQUIRED_OUTPUTS)
    assert manifest["prefix"] == expected_prefix
    assert manifest["manifest_key"] == "datasets/preprocessed/latest.json"
    assert len(client.objects) == 1
    assert client.objects[0]["Bucket"] == "private-bucket"
    assert client.objects[0]["Key"] == manifest["manifest_key"]
    uploaded_manifest = json.loads(client.objects[0]["Body"])
    assert uploaded_manifest["files"] == manifest["files"]


def test_upload_rejects_incomplete_preprocessed_output(tmp_path):
    client = FakeS3Client()

    with pytest.raises(FileNotFoundError, match="Output preprocessing tidak lengkap"):
        upload_preprocessed_artifact(
            s3_client=client,
            bucket="private-bucket",
            input_dir=tmp_path,
            run_id="123456789",
            commit_sha="abcdef0123456789",
        )

    assert client.uploaded == []
    assert client.objects == []
