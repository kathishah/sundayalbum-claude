"""Helper functions for API tests (not pytest fixtures)."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone

import boto3

REGION = "us-west-2"
SESSIONS_TABLE = "sa-sessions-test"
JOBS_TABLE = "sa-jobs-test"
USER_SETTINGS_TABLE = "sa-user-settings-test"
S3_BUCKET = "sa-data-test"


def make_session(email: str, user_hash: str) -> str:
    """Insert a valid session into DynamoDB and return the Bearer token."""
    token = str(uuid.uuid4())
    ddb = boto3.resource("dynamodb", region_name=REGION)
    ddb.Table(SESSIONS_TABLE).put_item(
        Item={
            "email": email,
            "user_hash": user_hash,
            "session_token": token,
            "token_expires_at": int(time.time()) + 7 * 24 * 3600,
            "verify_attempts": 0,
        }
    )
    return token


def make_event(
    route_key: str,
    body: dict | None = None,
    path_params: dict | None = None,
    token: str | None = None,
) -> dict:
    """Build a minimal API Gateway HTTP API event dict."""
    method = route_key.split()[0]
    return {
        "routeKey": route_key,
        "headers": {"authorization": f"Bearer {token}"} if token else {},
        "body": json.dumps(body) if body else None,
        "pathParameters": path_params or {},
        "requestContext": {"http": {"method": method}},
    }


def insert_job(
    user_hash: str,
    job_id: str,
    status: str = "complete",
    created_at: str | None = None,
    extra: dict | None = None,
) -> None:
    """Directly insert a job record into the moto DynamoDB jobs table."""
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()
    ddb = boto3.resource("dynamodb", region_name=REGION)
    item: dict = {
        "user_hash": user_hash,
        "job_id": job_id,
        "status": status,
        "current_step": "",
        "step_detail": "",
        "input_filename": "test.heic",
        "input_stem": "test",
        "upload_key": f"{user_hash}/uploads/test.heic",
        "photo_count": 1,
        "output_keys": [],
        "debug_keys": {},
        "thumbnail_keys": {},
        "created_at": created_at,
        "updated_at": created_at,
        "ttl": int(time.time()) + 30 * 24 * 3600,
        "execution_arn": "",
        "error_message": "",
        "processing_time": 0,
    }
    if extra:
        item.update(extra)
    ddb.Table(JOBS_TABLE).put_item(Item=item)
