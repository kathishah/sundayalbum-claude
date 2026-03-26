"""Shared utilities for Sunday Album Step Functions Lambda handlers.

Different from api/common.py (which serves HTTP API routes).
This module serves the pipeline step handlers invoked by Step Functions.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Environment ───────────────────────────────────────────────────────────────
S3_BUCKET = os.environ["S3_BUCKET"]
JOBS_TABLE = os.environ["JOBS_TABLE"]
REGION = os.environ.get("AWS_DEPLOY_REGION", "us-west-2")

# ── AWS clients ───────────────────────────────────────────────────────────────
_dynamodb = boto3.resource("dynamodb", region_name=REGION)
jobs_table = _dynamodb.Table(JOBS_TABLE)


# ── Storage factory ───────────────────────────────────────────────────────────

def make_storage(user_hash: str):
    """Return an S3Storage instance scoped to *user_hash*."""
    from src.storage.s3 import S3Storage
    return S3Storage(bucket=S3_BUCKET, prefix=user_hash, region=REGION)


# ── Pipeline config ───────────────────────────────────────────────────────────

def make_config(overrides: dict | None = None):
    """Return a PipelineConfig with optional overrides from the Step Functions event."""
    from src.pipeline import PipelineConfig
    cfg = PipelineConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


# ── DynamoDB helpers ──────────────────────────────────────────────────────────

def update_step(
    user_hash: str,
    job_id: str,
    step_name: str,
    detail: str = "",
    debug_keys: dict | None = None,
) -> None:
    """Update job record to reflect the currently running step."""
    now = datetime.now(timezone.utc).isoformat()
    update_expr = "SET current_step = :cs, step_detail = :sd, updated_at = :ua"
    expr_values: dict[str, Any] = {":cs": step_name, ":sd": detail, ":ua": now}

    if debug_keys:
        for attr_key, s3_key in debug_keys.items():
            safe = attr_key.replace("-", "_").replace(".", "_").replace("/", "_")
            update_expr += f", debug_keys.#dk_{safe} = :dkv_{safe}"
            expr_values[f":dkv_{safe}"] = s3_key

    try:
        kw: dict[str, Any] = {
            "Key": {"user_hash": user_hash, "job_id": job_id},
            "UpdateExpression": update_expr,
            "ExpressionAttributeValues": expr_values,
        }
        if debug_keys:
            kw["ExpressionAttributeNames"] = {
                f"#dk_{attr_key.replace('-','_').replace('.','_').replace('/','_')}": attr_key
                for attr_key in debug_keys
            }
        jobs_table.update_item(**kw)
    except Exception as exc:
        logger.warning("update_step DynamoDB error (non-fatal): %s", exc)


def finalize_job(
    user_hash: str,
    job_id: str,
    output_keys: list[str],
    processing_time: float = 0.0,
) -> None:
    """Mark job as complete with final output keys.

    output_keys are relative to the user_hash prefix (e.g. "output/Photo01.jpg").
    We store them with the user_hash prefix so api/jobs.py can generate correct
    presigned URLs directly: "{user_hash}/output/Photo01.jpg".
    """
    now = datetime.now(timezone.utc).isoformat()
    # Prefix keys with user_hash so full S3 paths are stored in DynamoDB
    full_keys = [
        k if k.startswith(f"{user_hash}/") else f"{user_hash}/{k}"
        for k in output_keys
    ]
    try:
        jobs_table.update_item(
            Key={"user_hash": user_hash, "job_id": job_id},
            UpdateExpression=(
                "SET #s = :s, current_step = :cs, step_detail = :sd, "
                "output_keys = :ok, photo_count = :pc, "
                "processing_time = :pt, updated_at = :ua"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "complete",
                ":cs": "done",
                ":sd": f"{len(full_keys)} photo(s) processed",
                ":ok": full_keys,
                ":pc": len(full_keys),
                ":pt": int(processing_time),
                ":ua": now,
            },
        )
        logger.info("Job %s marked complete (%d photos)", job_id, len(full_keys))
    except Exception as exc:
        logger.error("finalize_job DynamoDB error: %s", exc)


def fail_job(user_hash: str, job_id: str, error_message: str) -> None:
    """Mark job as failed."""
    now = datetime.now(timezone.utc).isoformat()
    try:
        jobs_table.update_item(
            Key={"user_hash": user_hash, "job_id": job_id},
            UpdateExpression="SET #s = :s, error_message = :em, updated_at = :ua",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": "failed",
                ":em": error_message[:2000],  # truncate long errors
                ":ua": now,
            },
        )
        logger.error("Job %s marked failed: %s", job_id, error_message[:200])
    except Exception as exc:
        logger.error("fail_job DynamoDB error: %s", exc)
