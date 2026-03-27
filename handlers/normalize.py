"""Step Functions handler: normalize."""
from __future__ import annotations
import logging
from typing import Any
from handlers.common import fail_job, make_config, make_storage, update_step
import src.steps.normalize as step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict, context: Any) -> dict:
    user_hash, job_id, stem = event["user_hash"], event["job_id"], event["stem"]
    update_step(user_hash, job_id, "normalize", "Resizing image")
    storage = make_storage(user_hash)
    config = make_config(event.get("config"))
    try:
        result = step.run(storage, stem, config)
    except Exception as exc:
        fail_job(user_hash, job_id, f"normalize failed: {exc}")
        raise
    update_step(
        user_hash, job_id, "normalize", "Normalized",
        debug_keys={"02_normalized": f"debug/{stem}_02_normalized.jpg"},
    )
    logger.info("normalize: %dx%d (scale=%.3f)", result["width"], result["height"], result["scale_factor"])
    return {**event, **result}
