"""OpenAI-based glare removal for the pipeline.

Adapts scripts/glare_remove.py into a pipeline-compatible module.
Works with numpy float32 RGB arrays [0, 1]; converts to/from temp PNG for the API.
"""

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Allowed API sizes per OpenAI docs
_ALLOWED_SIZES = {"1024x1024", "1536x1024", "1024x1536", "auto"}


def _pick_api_size(w: int, h: int) -> str:
    """Select the closest allowed API output size based on image orientation.

    Args:
        w: Image width in pixels
        h: Image height in pixels

    Returns:
        API size string: "1536x1024" (landscape), "1024x1536" (portrait), or "1024x1024" (square)
    """
    if w > h:
        return "1536x1024"
    if h > w:
        return "1024x1536"
    return "1024x1024"


def _build_prompt(scene_desc: str) -> str:
    """Build the glare removal prompt for the OpenAI edit endpoint.

    Args:
        scene_desc: One-sentence description of the scene in the photo.

    Returns:
        Full prompt string.
    """
    return (
        "We used an iPhone camera to photograph a picture printed on glossy paper for digitization. "
        "Remove glare/reflections caused by the glossy surface. "
        "Preserve the original composition, geometry, textures, and colors. "
        "Only modify pixels necessary to remove glare/reflections; do not change framing. "
        f"Description of the printed photo: {scene_desc}"
    )


def _numpy_to_png_bytes(image: np.ndarray) -> bytes:
    """Convert float32 RGB numpy array to PNG bytes.

    Args:
        image: float32 RGB [0, 1], shape (H, W, 3)

    Returns:
        PNG file bytes
    """
    uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(uint8, mode="RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL Image to float32 RGB numpy array.

    Args:
        pil_img: PIL Image (any mode)

    Returns:
        float32 RGB [0, 1], shape (H, W, 3)
    """
    rgb = np.array(pil_img.convert("RGB"))
    return rgb.astype(np.float32) / 255.0


def remove_glare_openai(
    image: np.ndarray,
    scene_desc: str,
    api_key: str,
    model: str = "gpt-image-1.5",
    quality: str = "high",
    input_fidelity: str = "high",
) -> np.ndarray:
    """Remove glare from a photo using OpenAI's image editing API.

    The input image is written to a temporary PNG, sent to the OpenAI
    ``images.edit`` endpoint with a glare-removal prompt, and the returned
    image is decoded back to a numpy array.

    On any API failure (network error, quota exceeded, invalid key, etc.) the
    original ``image`` is returned unchanged and a warning is logged so the
    pipeline never hard-fails on this step.

    Args:
        image: Input photo, float32 RGB [0, 1], shape (H, W, 3).
        scene_desc: One-sentence description of the scene (used in prompt).
        api_key: OpenAI API key.
        model: OpenAI image model to use. Default "gpt-image-1.5".
        quality: API quality parameter ("low", "medium", "high", "auto").
        input_fidelity: API input_fidelity parameter ("high" or "low").

    Returns:
        Glare-removed photo as float32 RGB [0, 1], shape may differ from input
        (API returns at most 1536×1024). Returns original image on failure.
    """
    h, w = image.shape[:2]
    api_size = _pick_api_size(w, h)
    prompt = _build_prompt(scene_desc)

    logger.debug(
        f"OpenAI glare removal: input={w}x{h}, api_size={api_size}, model={model}"
    )

    # Write numpy array to a temp PNG file (API requires a file path / file object)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(_numpy_to_png_bytes(image))

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        with tmp_path.open("rb") as f:
            resp = client.images.edit(
                model=model,
                image=f,
                prompt=prompt,
                n=1,
                size=api_size,
                quality=quality,
                input_fidelity=input_fidelity,
                output_format="png",
            )

        b64 = resp.data[0].b64_json
        img_bytes = base64.b64decode(b64)
        result_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = _pil_to_numpy(result_pil)

        out_h, out_w = result.shape[:2]
        logger.info(
            f"OpenAI glare removal complete: {w}x{h} → {out_w}x{out_h}"
        )
        return result

    except Exception as e:
        logger.warning(
            f"OpenAI glare removal failed, returning original image unchanged: {e}"
        )
        return image

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
