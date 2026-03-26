"""Step Functions handler: ai_orient (per-photo, runs inside Map state)."""
from __future__ import annotations
import logging
from typing import Any
from handlers.common import fail_job, make_config, make_storage, update_step
import src.steps.ai_orient as step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict, context: Any) -> dict:
    user_hash, job_id, stem = event["user_hash"], event["job_id"], event["stem"]
    photo_index: int = int(event["photo_index"])
    update_step(user_hash, job_id, "ai_orient", f"Orienting photo {photo_index}")
    storage = make_storage(user_hash)
    config = make_config(event.get("config"))
    try:
        result = step.run(storage, stem, config, photo_index=photo_index)
    except Exception as exc:
        fail_job(user_hash, job_id, f"ai_orient[{photo_index}] failed: {exc}")
        raise
    idx = f"{photo_index:02d}"
    update_step(
        user_hash, job_id, "ai_orient",
        f"Photo {photo_index} oriented ({result.get('rotation_degrees', 0)}°)",
        debug_keys={f"05b_photo_{idx}_oriented": f"debug/{stem}_05b_photo_{idx}_oriented.jpg"},
    )
    logger.info("ai_orient[%d]: %d° confidence=%s", photo_index, result.get("rotation_degrees", 0), result.get("orientation_confidence"))
    return {**event, **result}
