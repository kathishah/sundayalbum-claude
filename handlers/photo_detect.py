"""Step Functions handler: photo_detect."""
from __future__ import annotations
import logging
from typing import Any
from handlers.common import fail_job, make_config, make_storage, update_step
import src.steps.photo_detect as step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict, context: Any) -> dict:
    user_hash, job_id, stem = event["user_hash"], event["job_id"], event["stem"]
    update_step(user_hash, job_id, "photo_detect", "Detecting photos on page")
    storage = make_storage(user_hash)
    config = make_config(event.get("config"))
    try:
        result = step.run(storage, stem, config)
    except Exception as exc:
        fail_job(user_hash, job_id, f"photo_detect failed: {exc}")
        raise
    count = result.get("photo_count", 0)
    update_step(
        user_hash, job_id, "photo_detect", f"{count} photo(s) detected",
        debug_keys={"04_photo_boundaries": f"debug/{stem}_04_photo_boundaries.jpg"},
    )
    logger.info("photo_detect: %d photo(s)", count)
    return {**event, **result}
