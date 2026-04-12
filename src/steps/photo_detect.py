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

    # --- Forced-detections override ----------------------------------------
    # When the user has manually adjusted boundaries, the caller sets
    # config.forced_detections to a list of detection dicts.  Write them
    # directly to the JSON and return without running contour detection.
    if config.forced_detections is not None:
        # Normalise each dict: ensure required keys have defaults
        forced = [
            {
                "bbox": list(d["bbox"]),
                "confidence": float(d.get("confidence", 1.0)),
                "region_type": str(d.get("region_type", "photo")),
                "orientation": str(d.get("orientation", "unknown")),
            }
            for d in config.forced_detections
        ]
        result: dict = {
            "photo_count": len(forced),
            "multi_blob": False,
            "detections": forced,
        }
        storage.write_json(f"debug/{stem}_05_photo_detections.json", result)

        # Draw overlay — only if the page image is available (may be absent when
        # running photo_detect in isolation without preceding steps)
        try:
            page_image = storage.read_image(f"debug/{stem}_03_page_warped.jpg")
            overlay = _draw_forced_boundaries(page_image, forced)
            storage.write_image(
                f"debug/{stem}_04_photo_boundaries.jpg", overlay, format="jpeg", quality=90
            )
        except Exception as exc:
            logger.debug("photo_detect: could not write overlay for forced detections: %s", exc)

        logger.info("photo_detect: using %d forced detection(s)", len(forced))
        return result

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


def _draw_forced_boundaries(page_image: "np.ndarray", forced: list[dict]) -> "np.ndarray":
    """Draw coloured bounding boxes for forced detections onto *page_image*.

    Args:
        page_image: Float32 RGB [0, 1] page image.
        forced: List of detection dicts with ``bbox`` key.

    Returns:
        Float32 RGB [0, 1] overlay image.
    """
    import cv2
    import numpy as np

    colours = [
        (0.95, 0.65, 0.00),  # amber
        (0.18, 0.55, 0.34),  # green
        (0.24, 0.48, 0.78),  # blue
        (0.80, 0.25, 0.25),  # red
        (0.55, 0.25, 0.78),  # purple
    ]
    overlay = (page_image * 255).astype("uint8").copy()
    for i, det in enumerate(forced):
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        r, g, b = colours[i % len(colours)]
        colour_bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colour_bgr, 3)
        label = f"Photo {i + 1} (forced)"
        cv2.putText(overlay, label, (x1 + 6, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour_bgr, 2, cv2.LINE_AA)
    return overlay.astype("float32") / 255.0
