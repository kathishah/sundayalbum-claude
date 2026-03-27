"""Step: photo_detect — find individual photo boundaries on the page."""

import logging
from typing import Optional

from src.pipeline import PipelineConfig
from src.storage.backend import StorageBackend

logger = logging.getLogger(__name__)


def run(
    storage: StorageBackend,
    stem: str,
    config: PipelineConfig,
    photo_index: Optional[int] = None,
) -> dict:
    """Detect individual photo regions on the perspective-corrected page.

    Skips detection when the perspective step used the multi-blob path
    (photos were already extracted there).

    Reads:
    * ``debug/{stem}_03_page_warped.jpg``
    * ``debug/{stem}_03_page_detection.json``

    Writes:
    * ``debug/{stem}_04_photo_boundaries.jpg`` — detection overlay
    * ``debug/{stem}_05_photo_detections.json`` — detection results

    Args:
        storage: Active storage backend.
        stem: Input file stem.
        config: Pipeline configuration.
        photo_index: Unused; present for API uniformity.

    Returns:
        Dict with keys ``photo_count`` (int), ``multi_blob`` (bool),
        ``detections`` (list of detection dicts).
    """
    from src.photo_detection.detector import detect_photos, draw_photo_detections
    from src.page_detection.detector import draw_page_detection

    page_detection = storage.read_json(f"debug/{stem}_03_page_detection.json")
    multi_blob: bool = len(page_detection.get("photo_quads", [])) >= 2

    if multi_blob:
        blob_count = len(page_detection["photo_quads"])
        logger.info("photo_detect: multi-blob path — %d photos already extracted", blob_count)

        # Write a placeholder boundaries image (the page_warped image)
        page_image = storage.read_image(f"debug/{stem}_03_page_warped.jpg")
        storage.write_image(
            f"debug/{stem}_04_photo_boundaries.jpg",
            page_image,
            format="jpeg",
            quality=90,
        )

        result: dict = {
            "photo_count": blob_count,
            "multi_blob": True,
            "detections": [],
        }
        storage.write_json(f"debug/{stem}_05_photo_detections.json", result)
        return result

    # --- Normal contour detection ------------------------------------------
    page_image = storage.read_image(f"debug/{stem}_03_page_warped.jpg")
    page_was_corrected: bool = not page_detection["is_full_frame"]

    detections = detect_photos(
        page_image,
        min_area_ratio=config.photo_detect_min_area_ratio,
        max_count=config.photo_detect_max_count,
        method=config.photo_detect_method,
        page_was_corrected=page_was_corrected,
    )

    overlay = draw_photo_detections(page_image, detections)
    storage.write_image(
        f"debug/{stem}_04_photo_boundaries.jpg", overlay, format="jpeg", quality=90
    )

    # Serialise detections (bbox, confidence, region_type)
    det_list = [
        {
            "bbox": list(d.bbox),
            "confidence": float(d.confidence),
            "region_type": getattr(d, "region_type", "photo"),
            "orientation": getattr(d, "orientation", "unknown"),
        }
        for d in detections
    ]
    result = {
        "photo_count": len(detections),
        "multi_blob": False,
        "detections": det_list,
    }
    storage.write_json(f"debug/{stem}_05_photo_detections.json", result)

    logger.info("photo_detect: found %d photo(s)", len(detections))
    return result
