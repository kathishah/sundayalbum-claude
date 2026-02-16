"""Fade restoration for aged photos using CLAHE and saturation adjustment.

This module restores contrast and color vibrancy to faded photos using
Contrast Limited Adaptive Histogram Equalization (CLAHE) and selective
saturation boosting.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def restore_fading(
    photo: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_grid_size: tuple = (8, 8),
    saturation_boost: float = 0.15,
    auto_detect_fading: bool = True
) -> Tuple[np.ndarray, dict]:
    """Restore contrast and color to faded photos.

    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) on the L channel
    in LAB color space to restore local contrast. Optionally boosts saturation if
    the photo is detected to be faded.

    Args:
        photo: Input image as float32 RGB [0, 1], shape (H, W, 3)
        clahe_clip_limit: CLAHE clip limit (higher = more contrast, but more artifacts)
        clahe_grid_size: CLAHE grid size (smaller = more local, larger = more global)
        saturation_boost: Saturation multiplier (0.0 = no boost, 0.15 = 15% increase)
        auto_detect_fading: Only boost saturation if fading detected

    Returns:
        Tuple of:
            - Restored image as float32 RGB [0, 1]
            - Info dict with metrics about the restoration
    """
    if photo.ndim != 3 or photo.shape[2] != 3:
        logger.warning(f"Invalid photo shape: {photo.shape}, expected (H, W, 3)")
        return photo, {'error': 'invalid_shape'}

    # Assess fading before restoration
    fading_info_before = assess_fading(photo)
    logger.debug(
        f"Fading assessment before: contrast={fading_info_before['contrast_score']:.3f}, "
        f"saturation={fading_info_before['mean_saturation']:.3f}"
    )

    # Convert to LAB color space
    photo_uint8 = (photo * 255).astype(np.uint8)
    lab = cv2.cvtColor(photo_uint8, cv2.COLOR_RGB2LAB)

    # Extract L, a, b channels
    L, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (contrast restoration)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=clahe_grid_size
    )
    L_clahe = clahe.apply(L)

    # Measure contrast improvement
    contrast_before = _compute_local_contrast(L)
    contrast_after = _compute_local_contrast(L_clahe)
    contrast_improvement = contrast_after / (contrast_before + 1e-6)

    logger.debug(
        f"CLAHE applied: contrast improved by {contrast_improvement:.2f}x "
        f"(before={contrast_before:.3f}, after={contrast_after:.3f})"
    )

    # Merge back to LAB
    lab_clahe = cv2.merge([L_clahe, a, b])

    # Convert back to RGB
    rgb_clahe_uint8 = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    rgb_clahe = rgb_clahe_uint8.astype(np.float32) / 255.0

    # Saturation boost (if needed)
    saturation_boost_applied = 0.0
    if auto_detect_fading:
        # Only boost if fading detected
        if fading_info_before['is_faded']:
            # Adaptive boost: more boost for more faded photos
            fading_severity = fading_info_before['fading_score']
            adaptive_boost = saturation_boost * min(1.0, fading_severity / 0.3)
            rgb_restored = _boost_saturation(rgb_clahe, adaptive_boost)
            saturation_boost_applied = adaptive_boost
            logger.info(
                f"Fading detected (score={fading_severity:.3f}), "
                f"applying saturation boost: {adaptive_boost:.3f}"
            )
        else:
            rgb_restored = rgb_clahe
            logger.debug("No fading detected, skipping saturation boost")
    else:
        # Always apply fixed boost
        rgb_restored = _boost_saturation(rgb_clahe, saturation_boost)
        saturation_boost_applied = saturation_boost

    # Assess fading after restoration
    fading_info_after = assess_fading(rgb_restored)

    logger.info(
        f"Fade restoration complete: contrast {contrast_improvement:.2f}x, "
        f"saturation {fading_info_before['mean_saturation']:.3f} → "
        f"{fading_info_after['mean_saturation']:.3f}"
    )

    return rgb_restored, {
        'contrast_before': float(contrast_before),
        'contrast_after': float(contrast_after),
        'contrast_improvement': float(contrast_improvement),
        'saturation_before': fading_info_before['mean_saturation'],
        'saturation_after': fading_info_after['mean_saturation'],
        'saturation_boost_applied': float(saturation_boost_applied),
        'was_faded': fading_info_before['is_faded'],
        'fading_score_before': fading_info_before['fading_score'],
        'fading_score_after': fading_info_after['fading_score'],
    }


def _compute_local_contrast(gray_image: np.ndarray, kernel_size: int = 5) -> float:
    """Compute average local contrast in an image.

    Uses standard deviation in local neighborhoods as a measure of contrast.

    Args:
        gray_image: Grayscale image as uint8
        kernel_size: Size of local neighborhood

    Returns:
        Average local standard deviation (measure of contrast)
    """
    # Convert to float for computation
    img_float = gray_image.astype(np.float32)

    # Compute local mean
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    local_mean = cv2.filter2D(img_float, -1, kernel)

    # Compute local variance
    local_mean_sq = cv2.filter2D(img_float ** 2, -1, kernel)
    local_var = local_mean_sq - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Numerical stability

    # Standard deviation is square root of variance
    local_std = np.sqrt(local_var)

    # Average local std across the image
    avg_contrast = np.mean(local_std)

    return float(avg_contrast)


def _boost_saturation(photo: np.ndarray, boost: float) -> np.ndarray:
    """Boost color saturation in HSV space.

    Args:
        photo: Input image as float32 RGB [0, 1]
        boost: Saturation multiplier (e.g., 0.15 = increase by 15%)

    Returns:
        Image with boosted saturation as float32 RGB [0, 1]
    """
    if boost <= 0.0:
        return photo

    # Convert to HSV
    photo_uint8 = (photo * 255).astype(np.uint8)
    hsv = cv2.cvtColor(photo_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

    # Boost saturation channel
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + boost), 0, 255)

    # Convert back to RGB
    hsv_uint8 = hsv.astype(np.uint8)
    rgb_boosted_uint8 = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
    rgb_boosted = rgb_boosted_uint8.astype(np.float32) / 255.0

    return rgb_boosted


def assess_fading(photo: np.ndarray) -> dict:
    """Assess if a photo is faded and needs restoration.

    Checks both contrast (using local standard deviation) and color saturation.

    Args:
        photo: Input image as float32 RGB [0, 1]

    Returns:
        Dict with 'is_faded' (bool), 'fading_score' (0-1), 'contrast_score',
        'mean_saturation', and 'saturation_percentile_25'
    """
    # Convert to grayscale to measure contrast
    photo_uint8 = (photo * 255).astype(np.uint8)
    gray = cv2.cvtColor(photo_uint8, cv2.COLOR_RGB2GRAY)

    # Compute contrast
    contrast_score = _compute_local_contrast(gray, kernel_size=5)
    # Normalize: typical unfaded photos have contrast ~15-30
    normalized_contrast = contrast_score / 25.0

    # Compute saturation
    hsv = cv2.cvtColor(photo_uint8, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0

    mean_saturation = np.mean(saturation)
    saturation_p25 = np.percentile(saturation, 25)

    # Fading score: combination of low contrast and low saturation
    # Typical unfaded photos: contrast ~1.0, saturation ~0.3-0.5
    # Faded photos: contrast <0.6, saturation <0.2
    contrast_deficit = max(0.0, 0.8 - normalized_contrast)  # How much below normal
    saturation_deficit = max(0.0, 0.3 - mean_saturation)  # How much below normal

    fading_score = (contrast_deficit + saturation_deficit * 2) / 3.0  # Weight saturation more

    # Threshold: fading_score > 0.2 indicates noticeable fading
    is_faded = fading_score > 0.2

    logger.debug(
        f"Fading assessment: score={fading_score:.3f}, contrast={contrast_score:.2f}, "
        f"saturation={mean_saturation:.3f}, is_faded={is_faded}"
    )

    return {
        'is_faded': bool(is_faded),
        'fading_score': float(fading_score),
        'contrast_score': float(contrast_score),
        'normalized_contrast': float(normalized_contrast),
        'mean_saturation': float(mean_saturation),
        'saturation_percentile_25': float(saturation_p25),
    }


def restore_fading_conservative(photo: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Restore fading with conservative settings to avoid over-processing.

    Uses lower CLAHE clip limit and moderate saturation boost.
    Good for photos where you want subtle enhancement.

    Args:
        photo: Input image as float32 RGB [0, 1]

    Returns:
        Tuple of (restored image, info dict)
    """
    return restore_fading(
        photo,
        clahe_clip_limit=1.5,  # Lower than default 2.0
        clahe_grid_size=(8, 8),
        saturation_boost=0.10,  # Lower than default 0.15
        auto_detect_fading=True
    )


def restore_fading_aggressive(photo: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Restore fading with aggressive settings for heavily faded photos.

    Uses higher CLAHE clip limit and stronger saturation boost.
    Good for very old, severely faded photos.

    Args:
        photo: Input image as float32 RGB [0, 1]

    Returns:
        Tuple of (restored image, info dict)
    """
    return restore_fading(
        photo,
        clahe_clip_limit=3.0,  # Higher than default 2.0
        clahe_grid_size=(6, 6),  # Smaller grid = more local
        saturation_boost=0.25,  # Higher than default 0.15
        auto_detect_fading=True
    )
