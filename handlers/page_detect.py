"""Step Functions handler: page_detect."""
from __future__ import annotations
import logging
from typing import Any
from handlers.common import fail_job, make_config, make_storage, update_step
import src.steps.page_detect as step

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event: dict, context: Any) -> dict:
    user_hash, job_id, stem = event["user_hash"], event["job_id"], event["stem"]
    update_step(user_hash, job_id, "page_detect", "Detecting page boundary")
    storage = make_storage(user_hash)
    config = make_config(event.get("config"))
    try:
        result = step.run(storage, stem, config)
    except Exception as exc:
        fail_job(user_hash, job_id, f"page_detect failed: {exc}")
        raise
    update_step(
        user_hash, job_id, "page_detect", f"Page detected (confidence={result['confidence']:.2f})",
        debug_keys={"02_page_detected": f"debug/{stem}_02_page_detected.jpg"},
    )
    logger.info("page_detect: confidence=%.2f is_full_frame=%s", result["confidence"], result["is_full_frame"])
    return {**event, **result}
