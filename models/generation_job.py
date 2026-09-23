"""Mongo-backed persistence for durable asynchronous scenario-generation jobs."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import json
import uuid
from typing import Any

from gridfs import GridFSBucket
from pymongo import ReturnDocument

from models._helpers import collection_name
from models.database import get_database
from config.runtime import GENERATION_JOB_TTL_SECONDS


class GenerationJobModel:
    collection = get_database()[collection_name("MONGODB_GENERATION_JOBS_COLLECTION", "generation_jobs")]
    bucket_name = collection_name("MONGODB_GENERATION_RESULTS_BUCKET", "generation_results")

    @classmethod
    def ensure_indexes(cls) -> None:
        cls.collection.create_index("job_id", unique=True)
        cls.collection.create_index([("status", 1), ("created_at", 1)])
        cls.collection.create_index([("status", 1), ("lease_until", 1)])
        cls.collection.create_index("expires_at")

    @classmethod
    def create(cls, payload: dict[str, Any]) -> dict[str, Any]:
        job_id = f"gen-{uuid.uuid4().hex}"
        now = datetime.now(timezone.utc)
        document = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "expires_at": now + timedelta(seconds=GENERATION_JOB_TTL_SECONDS),
            "request": payload,
            "scenario_id": payload.get("scenario"),
            "draft_id": payload.get("draftId"),
            "count": int(payload.get("count", 0) or 0),
            "records_per_user": int(payload.get("recordsPerUser", 0) or 0),
            "attempts": 0,
        }
        cls.collection.insert_one(document)
        return {"job_id": job_id, **document}

    @classmethod
    def get(cls, job_id: str) -> dict[str, Any] | None:
        return cls.collection.find_one({"job_id": job_id}, {"_id": 0})

    @classmethod
    def claim_next(cls, worker_id: str, lease_seconds: int) -> dict[str, Any] | None:
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=max(30, lease_seconds))
        return cls.collection.find_one_and_update(
            {
                "$or": [
                    {"status": "queued"},
                    {"status": "running", "lease_until": {"$lt": now}},
                ],
                "expires_at": {"$gt": now},
            },
            {
                "$set": {
                    "status": "running",
                    "worker_id": worker_id,
                    "lease_until": lease_until,
                    "started_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempts": 1},
            },
            sort=[("created_at", 1)],
            return_document=ReturnDocument.AFTER,
            projection={"_id": 0},
        )

    @classmethod
    def heartbeat(cls, job_id: str, lease_seconds: int) -> bool:
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=max(30, lease_seconds))
        result = cls.collection.update_one(
            {"job_id": job_id, "status": "running"},
            {"$set": {"lease_until": lease_until, "updated_at": now}},
        )
        return result.modified_count == 1

    @classmethod
    def mark_completed(cls, job_id: str, result: dict[str, Any]) -> None:
        encoded = json.dumps(result, default=str, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        compressed = gzip.compress(encoded, compresslevel=6)
        bucket = GridFSBucket(get_database(), bucket_name=cls.bucket_name)
        stream_id = bucket.upload_from_stream(
            f"{job_id}.json.gz",
            compressed,
            metadata={"job_id": job_id, "content_type": "application/json", "encoding": "gzip"},
        )
        now = datetime.now(timezone.utc)
        cls.collection.update_one(
            {"job_id": job_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": now,
                    "updated_at": now,
                    "result_file_id": stream_id,
                    "result_bytes": len(encoded),
                    "result_compressed_bytes": len(compressed),
                    "result_summary": {
                        "scenario_id": result.get("scenario_id"),
                        "typeOfData": result.get("typeOfData"),
                        "total_records": result.get("total_records", 0),
                        "totalCount": result.get("totalCount", 0),
                        "fields": result.get("fields", []),
                        "validation_report": result.get("validation_report", {}),
                        "errors": result.get("errors", []),
                        "record_errors": result.get("record_errors", []),
                    },
                },
                "$unset": {"error": "", "lease_until": "", "worker_id": ""},
            },
        )

    @classmethod
    def result(cls, job_id: str) -> dict[str, Any] | None:
        row = cls.get(job_id)
        if not row or row.get("status") != "completed" or not row.get("result_file_id"):
            return None
        bucket = GridFSBucket(get_database(), bucket_name=cls.bucket_name)
        data = bucket.open_download_stream(row["result_file_id"]).read()
        return json.loads(gzip.decompress(data).decode("utf-8"))

    @classmethod
    def mark_failed_or_retry(cls, job_id: str, error: str, max_attempts: int) -> bool:
        now = datetime.now(timezone.utc)
        current = cls.collection.find_one({"job_id": job_id}, {"attempts": 1, "_id": 0}) or {}
        attempts = int(current.get("attempts", 0) or 0)
        if attempts < max(1, max_attempts):
            result = cls.collection.update_one(
                {"job_id": job_id, "status": "running"},
                {
                    "$set": {"status": "queued", "updated_at": now, "last_error": str(error)[:10000]},
                    "$unset": {"lease_until": "", "worker_id": "", "started_at": ""},
                },
            )
            return result.modified_count == 1
        cls.collection.update_one(
            {"job_id": job_id, "status": "running"},
            {
                "$set": {
                    "status": "failed",
                    "failed_at": now,
                    "updated_at": now,
                    "error": str(error)[:10000],
                },
                "$unset": {"lease_until": "", "worker_id": ""},
            },
        )
        return False

    @classmethod
    def cleanup_expired_jobs(cls) -> int:
        """Delete expired jobs and their GridFS result files."""
        now = datetime.now(timezone.utc)
        expired = list(cls.collection.find({"expires_at": {"$lte": now}}, {"_id": 0, "job_id": 1, "result_file_id": 1}))
        if not expired:
            return 0
        bucket = GridFSBucket(get_database(), bucket_name=cls.bucket_name)
        for item in expired:
            file_id = item.get("result_file_id")
            if file_id is not None:
                try:
                    bucket.delete(file_id)
                except Exception:
                    pass
        result = cls.collection.delete_many({"expires_at": {"$lte": now}})
        return int(result.deleted_count)
