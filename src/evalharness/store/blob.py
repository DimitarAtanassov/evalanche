"""Blob storage abstraction."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import boto3
from botocore.client import BaseClient

from evalharness.config import get_settings
from evalharness.hashing import sha256_hex


class BlobStore(ABC):
    @abstractmethod
    async def put_json(self, key: str, data: dict[str, Any]) -> str: ...

    @abstractmethod
    async def get_json(self, uri: str) -> dict[str, Any]: ...


class FilesystemBlobStore(BlobStore):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    async def put_json(self, key: str, data: dict[str, Any]) -> str:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return f"file://{path}"

    async def get_json(self, uri: str) -> dict[str, Any]:
        path = Path(uri.removeprefix("file://"))
        return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


class MinioBlobStore(BlobStore):
    def __init__(self, client: BaseClient, bucket: str) -> None:
        self.client = client
        self.bucket = bucket

    async def put_json(self, key: str, data: dict[str, Any]) -> str:
        body = json.dumps(data).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        return f"s3://{self.bucket}/{key}"

    async def get_json(self, uri: str) -> dict[str, Any]:
        _, _, bucket_key = uri.partition("s3://")
        bucket, _, key = bucket_key.partition("/")
        obj = self.client.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))  # type: ignore[no-any-return]


def get_blob_store() -> BlobStore:
    settings = get_settings()
    if settings.blob_storage_backend == "filesystem":
        return FilesystemBlobStore(Path(settings.filesystem_blob_root))
    client = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )
    return MinioBlobStore(client, settings.minio_bucket)


def blob_key_for_raw(run_id: str, case_external_id: str, repeat_idx: int) -> str:
    digest = sha256_hex(f"{run_id}:{case_external_id}:{repeat_idx}")
    return f"raw/{run_id}/{digest}.json"
