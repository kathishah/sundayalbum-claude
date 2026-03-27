"""Step Functions handler: photo_split."""
from __future__ import annotations
import logging
from typing import Any
from handlers.common import fail_job, make_config, make_storage, update_step
import src.steps.photo_split as step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict, context: Any) -> dict:
    user_hash, job_id, stem = event["user_hash"], event["job_id"], event["stem"]
    update_step(user_hash, job_id, "photo_split", "Splitting individual photos")
    storage = make_storage(user_hash)
    config = make_config(event.get("config"))
    try:
        result = step.run(storage, stem, config)
    except Exception as exc:
        fail_job(user_hash, job_id, f"photo_split failed: {exc}")
        raise
    count = result.get("photo_count", 1)
    update_step(user_hash, job_id, "photo_split", f"Split into {count} photo(s)")
    logger.info("photo_split: %d photo(s)", count)
    # photo_count must be at the top level for the Step Functions PrepareMap pass state
    return {**event, "photo_count": count}
