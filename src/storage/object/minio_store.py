"""
storage/object/minio_store.py
──────────────────────────────
MinIO (S3-compatible) object store for raw source files.
Files are stored here permanently for audit and re-ingestion.
"""
from __future__ import annotations

import io
from typing import Optional

from minio import Minio
from minio.error import S3Error

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class MinioStore:

    def __init__(self):
        self._client: Optional[Minio] = None

    def _get_client(self) -> Minio:
        if self._client is None:
            self._client = Minio(
                endpoint=settings.minio_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )
        return self._client

    async def upload(self, file_bytes: bytes, filename: str, source_id: str) -> str:
        """Upload file to MinIO. Returns the object URL."""
        client = self._get_client()
        object_key = f"{source_id}/{filename}"
        bucket = settings.minio_bucket_documents

        try:
            client.put_object(
                bucket_name=bucket,
                object_name=object_key,
                data=io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type=self._guess_content_type(filename),
            )
            url = f"minio://{bucket}/{object_key}"
            logger.info("File uploaded", object_key=object_key, size=len(file_bytes))
            return url
        except S3Error as e:
            logger.error("MinIO upload failed", object_key=object_key, error=str(e))
            raise

    async def download(self, object_key: str) -> bytes:
        """Download file from MinIO by object key."""
        client = self._get_client()
        bucket = settings.minio_bucket_documents
        try:
            response = client.get_object(bucket, object_key)
            data = response.read()
            response.close()
            return data
        except S3Error as e:
            logger.error("MinIO download failed", object_key=object_key, error=str(e))
            raise

    def get_presigned_url(self, object_key: str, expires_hours: int = 1) -> str:
        """Generate a presigned URL for direct file access (audit, download)."""
        from datetime import timedelta
        client = self._get_client()
        return client.presigned_get_object(
            settings.minio_bucket_documents,
            object_key,
            expires=timedelta(hours=expires_hours),
        )

    def _guess_content_type(self, filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower()
        return {
            "pdf": "application/pdf",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "csv": "text/csv",
            "txt": "text/plain",
            "md": "text/markdown",
        }.get(ext, "application/octet-stream")


_minio: Optional[MinioStore] = None

def get_minio_store() -> MinioStore:
    global _minio
    if _minio is None:
        _minio = MinioStore()
    return _minio