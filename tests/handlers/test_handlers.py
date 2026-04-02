"""Tests 23–35: Pipeline handler write-correctness, skip logic, finalize."""

from __future__ import annotations

import io
import time
import uuid
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pytest
from PIL import Image

from handler_helpers import (
    FAKE_IMAGE,
    JOBS_TABLE,
    REGION,
    S3_BUCKET,
    create_job_record,
    get_job,
    make_base_event,
    put_s3_image,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ids() -> tuple[str, str]:
    """Return a fresh (user_hash, job_id) pair."""
    return f"user-{uuid.uuid4().hex[:8]}", str(uuid.uuid4())


def _jpeg_bytes() -> bytes:
    img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── 23. load writes debug_keys and thumbnail_keys ────────────────────────────


def test_load_writes_debug_and_thumbnail_keys():
    """load.handler writes 01_loaded to both debug_keys and thumbnail_keys in DDB."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)

    # Put a real JPEG at the upload key so load.handler can read it
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=S3_BUCKET,
        Key=f"{user_hash}/uploads/test.heic",
        Body=_jpeg_bytes(),
    )

    # Patch load_image so we don't need a real HEIC decoder
    mock_meta = MagicMock()
    mock_meta.original_size = (4, 4)
    mock_meta.format = "HEIC"

    with patch("src.preprocessing.loader.load_image", return_value=(FAKE_IMAGE, mock_meta)):
        from handlers import load as load_handler
        event = make_base_event(user_hash, job_id)
        load_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "01_loaded" in item["debug_keys"]
    assert "01_loaded" in item["thumbnail_keys"]

    # Verify S3 objects exist
    s3 = boto3.client("s3", region_name=REGION)
    debug_key = item["debug_keys"]["01_loaded"]
    s3.head_object(Bucket=S3_BUCKET, Key=f"{user_hash}/{debug_key}")
    thumb_key = item["thumbnail_keys"]["01_loaded"]
    s3.head_object(Bucket=S3_BUCKET, Key=f"{user_hash}/{thumb_key}")


# ── 24. normalize writes keys ────────────────────────────────────────────────


def test_normalize_writes_keys():
    """normalize.handler writes 02_normalized to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_01_loaded.jpg")

    with patch("src.steps.normalize.run", return_value={"width": 4, "height": 4, "scale_factor": 1.0}):
        from handlers import normalize as normalize_handler
        event = make_base_event(user_hash, job_id)
        normalize_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "02_normalized" in item["debug_keys"]
    assert "02_normalized" in item["thumbnail_keys"]


# ── 25. page_detect writes keys ──────────────────────────────────────────────


def test_page_detect_writes_keys():
    """page_detect.handler writes 02_page_detected to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_01_loaded.jpg")
    put_s3_image(user_hash, "debug/test_02_normalized.jpg")

    with patch("src.steps.page_detect.run", return_value={"detected": True, "confidence": 0.9, "is_full_frame": False}):
        from handlers import page_detect as page_detect_handler
        event = make_base_event(user_hash, job_id)
        page_detect_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "02_page_detected" in item["debug_keys"]
    assert "02_page_detected" in item["thumbnail_keys"]


# ── 26. perspective writes keys ───────────────────────────────────────────────


def test_perspective_writes_keys():
    """perspective.handler writes 03_page_warped to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_02_normalized.jpg")

    with patch("src.steps.perspective.run", return_value={"warped": True}):
        from handlers import perspective as perspective_handler
        event = make_base_event(user_hash, job_id)
        perspective_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "03_page_warped" in item["debug_keys"]
    assert "03_page_warped" in item["thumbnail_keys"]


# ── 27. photo_split writes per-photo keys ────────────────────────────────────


def test_photo_split_writes_keys_per_photo():
    """photo_split.handler with a 2-photo result writes photo_01 and photo_02 keys."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_03_page_warped.jpg")

    # Simulate 2 photos detected
    with patch(
        "src.steps.photo_split.run",
        return_value={"photo_count": 2, "photo_indices": [1, 2]},
    ):
        from handlers import photo_split as photo_split_handler
        event = {**make_base_event(user_hash, job_id), "photo_count": 2}
        photo_split_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "05_photo_01_raw" in item["debug_keys"]
    assert "05_photo_02_raw" in item["debug_keys"]
    assert "05_photo_01_raw" in item["thumbnail_keys"]
    assert "05_photo_02_raw" in item["thumbnail_keys"]


# ── 28. ai_orient writes keys ────────────────────────────────────────────────


def test_ai_orient_writes_keys():
    """ai_orient.handler writes 05b_photo_01_oriented to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_05_photo_01_raw.jpg")

    with patch(
        "src.steps.ai_orient.run",
        return_value={
            "rotation_degrees": 0,
            "flip_horizontal": False,
            "scene_description": "test",
        },
    ):
        from handlers import ai_orient as ai_orient_handler
        event = {**make_base_event(user_hash, job_id), "photo_index": 1}
        ai_orient_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "05b_photo_01_oriented" in item["debug_keys"]
    assert "05b_photo_01_oriented" in item["thumbnail_keys"]


# ── 29. glare_remove writes keys ─────────────────────────────────────────────


def test_glare_remove_writes_keys():
    """glare_remove.handler writes 07_photo_01_deglared to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_05b_photo_01_oriented.jpg")

    with patch("src.steps.glare_remove.run", return_value={"removed": True}):
        from handlers import glare_remove as glare_remove_handler
        event = {**make_base_event(user_hash, job_id), "photo_index": 1}
        glare_remove_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "07_photo_01_deglared" in item["debug_keys"]
    assert "07_photo_01_deglared" in item["thumbnail_keys"]


# ── 30. color_restore writes keys ────────────────────────────────────────────


def test_color_restore_writes_keys():
    """color_restore.handler writes 14_photo_01_enhanced to both key maps."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_07_photo_01_deglared.jpg")

    with patch("src.steps.color_restore.run", return_value={"enhanced": True}):
        from handlers import color_restore as color_restore_handler
        event = {**make_base_event(user_hash, job_id), "photo_index": 1}
        color_restore_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert "14_photo_01_enhanced" in item["debug_keys"]
    assert "14_photo_01_enhanced" in item["thumbnail_keys"]


# ── 31. step failure marks job failed ────────────────────────────────────────


def test_step_failure_marks_job_failed():
    """An exception inside a step sets DDB status='failed' with a non-empty error_message."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)
    put_s3_image(user_hash, "debug/test_01_loaded.jpg")

    with patch("src.steps.normalize.run", side_effect=RuntimeError("disk full")):
        from handlers import normalize as normalize_handler
        event = make_base_event(user_hash, job_id)
        with pytest.raises(RuntimeError):
            normalize_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert item["status"] == "failed"
    assert item["error_message"] != ""


# ── 32. skip_pre_split when start_from is a later step ───────────────────────


def test_skip_pre_split_when_start_from_later():
    """should_skip_pre_split returns True for 'load' when start_from='ai_orient'."""
    from handlers.common import should_skip_pre_split

    event_reprocess = {**make_base_event("u", "j"), "start_from": "ai_orient"}
    assert should_skip_pre_split(event_reprocess, "load") is True
    assert should_skip_pre_split(event_reprocess, "normalize") is True
    assert should_skip_pre_split(event_reprocess, "page_detect") is True


# ── 33. skip per-photo when index doesn't match ───────────────────────────────


def test_skip_per_photo_wrong_index():
    """should_skip_per_photo returns True when reprocess_photo_index != photo_index."""
    from handlers.common import should_skip_per_photo

    base = make_base_event("u", "j")

    # Photo 2 should be skipped (reprocess_photo_index=1, photo_index=2 in event)
    event_photo2 = {**base, "start_from": "ai_orient", "reprocess_photo_index": 1, "photo_index": 2}
    assert should_skip_per_photo(event_photo2, "ai_orient") is True

    # Photo 1 should NOT be skipped
    event_photo1 = {**base, "start_from": "ai_orient", "reprocess_photo_index": 1, "photo_index": 1}
    assert should_skip_per_photo(event_photo1, "ai_orient") is False


# ── 34. finalize marks job complete ──────────────────────────────────────────


def test_finalize_marks_complete():
    """finalize.handler sets status='complete', writes output_keys, sets photo_count."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)

    output_key = f"{user_hash}/output/SundayAlbum_test_Photo01.jpg"
    boto3.client("s3", region_name=REGION).put_object(
        Bucket=S3_BUCKET, Key=output_key, Body=b"output"
    )

    from handlers import finalize as finalize_handler
    event = {
        **make_base_event(user_hash, job_id),
        "photo_results": [{"photo_index": 1, "output_key": output_key}],
        "start_time": time.time() - 10,
    }
    result = finalize_handler.handler(event, None)

    assert result["status"] == "complete"
    assert result["photo_count"] == 1

    item = get_job(user_hash, job_id)
    assert item["status"] == "complete"
    assert output_key in item["output_keys"]
    assert item["photo_count"] == 1


# ── 35. finalize with empty results marks failed ─────────────────────────────


def test_finalize_empty_results_marks_failed():
    """finalize.handler with no photo_results raises and marks the job failed."""
    user_hash, job_id = _ids()
    create_job_record(user_hash, job_id)

    from handlers import finalize as finalize_handler
    event = {
        **make_base_event(user_hash, job_id),
        "photo_results": [],
        "start_time": time.time() - 5,
    }
    with pytest.raises(RuntimeError):
        finalize_handler.handler(event, None)

    item = get_job(user_hash, job_id)
    assert item["status"] == "failed"
