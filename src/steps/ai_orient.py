"""Step: ai_orient — Claude Haiku orientation correction per photo."""

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
    """Correct gross rotation errors (90°/180°/270°) with Claude Haiku.

    One API call per photo.  Also produces a ``scene_description`` used by
    the glare-removal step to improve prompt quality.

    If ``ANTHROPIC_API_KEY`` is absent or AI orientation is disabled in
    *config*, the input image is copied through unchanged.

    Reads ``debug/{stem}_05_photo_NN_raw.jpg`` (where NN = *photo_index*).
    Writes:
    * ``debug/{stem}_05b_photo_NN_oriented.jpg``
    * ``debug/{stem}_05b_photo_NN_analysis.json``

    Args:
        storage: Active storage backend.
        stem: Input file stem.
        config: Pipeline configuration.
        photo_index: 1-based photo index (required for this step).

    Returns:
        Dict with ``rotation_degrees``, ``flip_horizontal``,
        ``orientation_confidence``, ``scene_description``.
    """
    if photo_index is None:
        raise ValueError("ai_orient.run() requires photo_index")

    idx = f"{photo_index:02d}"
    in_key = f"debug/{stem}_05_photo_{idx}_raw.jpg"
    out_key = f"debug/{stem}_05b_photo_{idx}_oriented.jpg"
    analysis_key = f"debug/{stem}_05b_photo_{idx}_analysis.json"

    image = storage.read_image(in_key)

    if not config.use_ai_orientation:
        logger.info("ai_orient[%d]: disabled — passing through", photo_index)
        storage.write_image(out_key, image, format="jpeg", quality=95)
        result = {
            "rotation_degrees": 0,
            "flip_horizontal": False,
            "orientation_confidence": "skipped",
            "scene_description": "",
        }
        storage.write_json(analysis_key, result)
        return result

    from src.ai.claude_vision import analyze_photo_for_processing, apply_orientation
    from src.utils.secrets import load_secrets

    secrets = load_secrets()
    api_key = secrets.anthropic_api_key

    if not api_key:
        logger.warning("ai_orient[%d]: ANTHROPIC_API_KEY not set — passing through", photo_index)
        storage.write_image(out_key, image, format="jpeg", quality=95)
        result = {
            "rotation_degrees": 0,
            "flip_horizontal": False,
            "orientation_confidence": "skipped",
            "scene_description": "",
        }
        storage.write_json(analysis_key, result)
        return result

    analysis = analyze_photo_for_processing(
        image, api_key=api_key, model=config.ai_orientation_model
    )

    if analysis.orientation_confidence in ("medium", "high"):
        corrected = apply_orientation(image, analysis)
        logger.info(
            "ai_orient[%d]: %d°, flip=%s, confidence=%s",
            photo_index,
            analysis.rotation_degrees,
            analysis.flip_horizontal,
            analysis.orientation_confidence,
        )
    else:
        corrected = image
        logger.debug(
            "ai_orient[%d]: confidence=%s — not applying correction",
            photo_index,
            analysis.orientation_confidence,
        )

    storage.write_image(out_key, corrected, format="jpeg", quality=95)

    result = {
        "rotation_degrees": int(analysis.rotation_degrees),
        "flip_horizontal": bool(analysis.flip_horizontal),
        "orientation_confidence": str(analysis.orientation_confidence),
        "scene_description": str(analysis.scene_description or ""),
    }
    storage.write_json(analysis_key, result)
    return result
