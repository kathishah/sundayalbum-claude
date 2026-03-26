"""Step: photo_split — extract individual photos from the page image."""

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
    """Extract individual photo crops from the page and write them to storage.

    For the multi-blob path: renames the already-extracted blob images
    (``debug/{stem}_03b_blob_NN_extracted.jpg``) to the standard per-photo
    key (``debug/{stem}_05_photo_NN_raw.jpg``).

    For the normal path: uses the detected bounding boxes to crop each photo
    from the page image and writes each crop.

    Falls back to writing the full page as a single photo if no detections.

    Reads:
    * ``debug/{stem}_03_page_warped.jpg``
    * ``debug/{stem}_05_photo_detections.json``
    * ``debug/{stem}_03b_blob_NN_extracted.jpg`` (multi-blob only)

    Writes:
    * ``debug/{stem}_05_photo_NN_raw.jpg`` for each photo (1-indexed)

    Args:
        storage: Active storage backend.
        stem: Input file stem.
        config: Pipeline configuration.
        photo_index: Unused; present for API uniformity.

    Returns:
        Dict with ``photo_count`` (int).
    """
    from src.photo_detection.splitter import split_photos
    from src.photo_detection.detector import detect_photos

    detection_result = storage.read_json(f"debug/{stem}_05_photo_detections.json")
    multi_blob: bool = detection_result.get("multi_blob", False)
    photo_count: int = detection_result["photo_count"]

    # --- Multi-blob: blob images already extracted by perspective step ------
    if multi_blob:
        logger.info("photo_split: multi-blob — copying %d blob images", photo_count)
        for i in range(1, photo_count + 1):
            blob_image = storage.read_image(f"debug/{stem}_03b_blob_{i:02d}_extracted.jpg")
            storage.write_image(
                f"debug/{stem}_05_photo_{i:02d}_raw.jpg",
                blob_image,
                format="jpeg",
                quality=95,
            )
        return {"photo_count": photo_count}

    # --- Normal path: split from page image --------------------------------
    page_image = storage.read_image(f"debug/{stem}_03_page_warped.jpg")

    if photo_count == 0:
        # No detections — treat entire page as one photo
        logger.info("photo_split: no detections — treating full page as single photo")
        storage.write_image(
            f"debug/{stem}_05_photo_01_raw.jpg", page_image, format="jpeg", quality=95
        )
        return {"photo_count": 1}

    # Rebuild PhotoDetection objects from stored JSON
    # Re-run detection on the same image to get proper objects for splitter
    det_dicts = detection_result["detections"]

    # We stored bbox + region_type; reconstruct minimal objects for split_photos
    # by re-running detect_photos (idempotent on the same image+config)
    page_detection_json = storage.read_json(f"debug/{stem}_03_page_detection.json")
    page_was_corrected: bool = not page_detection_json["is_full_frame"]

    detections = detect_photos(
        page_image,
        min_area_ratio=config.photo_detect_min_area_ratio,
        max_count=config.photo_detect_max_count,
        method=config.photo_detect_method,
        page_was_corrected=page_was_corrected,
    )

    if not detections:
        logger.warning("photo_split: re-detection returned 0 photos — using full page")
        storage.write_image(
            f"debug/{stem}_05_photo_01_raw.jpg", page_image, format="jpeg", quality=95
        )
        return {"photo_count": 1}

    extracted = split_photos(page_image, detections)

    for i, photo in enumerate(extracted, 1):
        storage.write_image(
            f"debug/{stem}_05_photo_{i:02d}_raw.jpg",
            photo,
            format="jpeg",
            quality=95,
        )
        logger.debug(
            "photo_split: wrote photo %d (%dx%d)", i, photo.shape[1], photo.shape[0]
        )

    actual_count = len(extracted)
    logger.info("photo_split: extracted %d photo(s)", actual_count)
    return {"photo_count": actual_count}
