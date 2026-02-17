"""Detect album page boundaries in images.

Handles two categories of input:
- Album pages (multi-photo): detects the album page boundary
- Individual prints: detects the photo print edges

Three detection algorithms are available:
- "edge": Canny edges + contour/Hough fitting (fast, ~0.1s). Rejects full-frame
  false positives via max_area_ratio to avoid detecting the background surface
  instead of the actual subject.
- "color": Background color segmentation via corner sampling (medium, ~0.2s).
  Samples image corners to build a LAB-space background model, then segments
  the foreground. Works when the background has a clearly different color from
  the subject (e.g., table surface vs. photo print).
- "grabcut": GrabCut iterative foreground/background segmentation (slowest, ~1-2s).
  Most robust for difficult cases. Initializes with a slightly-inset rectangle.
- "auto": Tries edge → color → grabcut in sequence, returns first success.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum area ratio for edge-based detection.
# Contours larger than this are assumed to be the background-surface perimeter,
# not the actual subject. Tuned to reject 0.97-0.99 false positives while keeping
# legitimately near-full-frame subjects (album pages at ~0.90-0.93).
_EDGE_MAX_AREA_RATIO = 0.95

# Maximum area ratio for mask-based methods (color, grabcut).
# These don't have the same full-frame false positive problem, so we're permissive.
_MASK_MAX_AREA_RATIO = 0.99


@dataclass
class PageDetection:
    """Result of page boundary detection."""

    corners: np.ndarray  # shape (4, 2) — four corner points as (x, y)
    confidence: float    # 0.0 to 1.0
    is_full_frame: bool  # True if no clear boundary found, using full image
    method_used: str = field(default="none")  # Which sub-strategy succeeded
    area_ratio: float = field(default=0.0)    # Detected quad area / total image area


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        Ordered array of shape (4, 2).
    """
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[1] = pts[np.argmin(d)]   # top-right
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    ordered[3] = pts[np.argmax(d)]   # bottom-left

    return ordered


def _find_largest_quadrilateral(
    edges: np.ndarray,
    image_area: float,
    min_area_ratio: float,
    max_area_ratio: float = _EDGE_MAX_AREA_RATIO,
) -> Optional[np.ndarray]:
    """Find the largest quadrilateral contour within the area bounds.

    Args:
        edges: Binary edge image from Canny or threshold.
        image_area: Total image area (height * width).
        min_area_ratio: Minimum contour area as fraction of image area.
        max_area_ratio: Maximum contour area as fraction of image area.
            Contours above this are skipped as background-perimeter false positives.

    Returns:
        Ordered corner points as (4, 2) array, or None if not found.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)
        area_ratio = area / image_area

        if area_ratio < min_area_ratio:
            break  # Sorted descending; everything below is also too small

        if area_ratio > max_area_ratio:
            logger.debug(
                f"Skipping contour: area_ratio={area_ratio:.3f} > "
                f"max_area_ratio={max_area_ratio:.3f} (likely background perimeter)"
            )
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

    return None


def _find_quad_via_hough(
    edges: np.ndarray,
    image_shape: tuple,
    min_area_ratio: float,
    max_area_ratio: float = _EDGE_MAX_AREA_RATIO,
) -> Optional[np.ndarray]:
    """Find page quadrilateral using Hough line detection.

    Fallback when contour-based detection doesn't find a clean quad.

    Args:
        edges: Binary edge image.
        image_shape: (height, width) of the image.
        min_area_ratio: Minimum area ratio for valid detection.
        max_area_ratio: Maximum area ratio for valid detection.

    Returns:
        Ordered corner points as (4, 2) array, or None if not found.
    """
    height, width = image_shape[:2]
    image_area = height * width

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min(height, width) * 0.15,
        maxLineGap=20,
    )

    if lines is None or len(lines) < 4:
        return None

    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))

        if angle < 30:
            horizontal_lines.append(line[0])
        elif angle > 60:
            vertical_lines.append(line[0])

    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None

    h_lines = np.array(horizontal_lines)
    v_lines = np.array(vertical_lines)

    h_y_mids = (h_lines[:, 1] + h_lines[:, 3]) / 2
    h_sorted_idx = np.argsort(h_y_mids)
    top_line = h_lines[h_sorted_idx[0]]
    bottom_line = h_lines[h_sorted_idx[-1]]

    v_x_mids = (v_lines[:, 0] + v_lines[:, 2]) / 2
    v_sorted_idx = np.argsort(v_x_mids)
    left_line = v_lines[v_sorted_idx[0]]
    right_line = v_lines[v_sorted_idx[-1]]

    def line_intersection(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float32)

    corners = []
    for h_line, v_line in [
        (top_line, left_line),
        (top_line, right_line),
        (bottom_line, right_line),
        (bottom_line, left_line),
    ]:
        pt = line_intersection(h_line, v_line)
        if pt is None:
            return None
        corners.append(pt)

    corners_arr = np.array(corners, dtype=np.float32)

    margin = max(height, width) * 0.1
    if (np.any(corners_arr < -margin)
            or np.any(corners_arr[:, 0] > width + margin)
            or np.any(corners_arr[:, 1] > height + margin)):
        return None

    corners_arr[:, 0] = np.clip(corners_arr[:, 0], 0, width - 1)
    corners_arr[:, 1] = np.clip(corners_arr[:, 1], 0, height - 1)

    ordered = _order_corners(corners_arr)
    quad_area = cv2.contourArea(ordered.astype(np.int32))
    area_ratio = quad_area / image_area

    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None

    return ordered


def _mask_to_quad(
    mask: np.ndarray,
    image_h: int,
    image_w: int,
    image_area: float,
    min_area_ratio: float,
    max_area_ratio: float = _MASK_MAX_AREA_RATIO,
) -> Optional[np.ndarray]:
    """Convert a binary foreground mask to a quadrilateral boundary.

    Cleans the mask with morphological ops, finds the largest foreground region,
    and returns its minimum-area bounding rectangle as 4 corners.

    Args:
        mask: Binary mask (uint8, 0 or 255) — 255 is foreground.
        image_h: Image height in pixels.
        image_w: Image width in pixels.
        image_area: Total image area (h * w).
        min_area_ratio: Minimum foreground area as fraction of image area.
        max_area_ratio: Maximum foreground area as fraction of image area.

    Returns:
        Ordered corner points as (4, 2) float32 array, or None if not found.
    """
    # Morphological close to fill holes (e.g., glare patches inside the subject),
    # then open to remove small noise. Kernel proportional to image size.
    kernel_size = max(15, int(min(image_h, image_w) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / image_area

    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        logger.debug(
            f"Mask-to-quad: rejected contour (area_ratio={area_ratio:.3f}, "
            f"bounds=[{min_area_ratio:.2f}, {max_area_ratio:.2f}])"
        )
        return None

    # Use convex hull for cleaner corner estimation
    hull = cv2.convexHull(largest)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    corners = np.array(box, dtype=np.float32)

    corners[:, 0] = np.clip(corners[:, 0], 0, image_w - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, image_h - 1)

    return _order_corners(corners)


def _find_boundary_via_bg_color(
    image: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float = _MASK_MAX_AREA_RATIO,
) -> Optional[np.ndarray]:
    """Detect subject boundary by sampling background color from image corners.

    Samples four corner patches to build a statistical background color model
    in LAB space, then classifies each pixel by its Mahalanobis-like distance
    from that model.

    Works best when the subject (print/album page) has a clearly different color
    from the background (table surface, desk, etc.) and the corners of the image
    contain background pixels rather than subject pixels.

    Args:
        image: Input image as float32 RGB [0, 1].
        min_area_ratio: Minimum valid subject area as fraction of total image area.
        max_area_ratio: Maximum valid subject area as fraction of total image area.

    Returns:
        Ordered corner points as (4, 2) float32 array, or None if not found.
    """
    h, w = image.shape[:2]
    image_area = float(h * w)

    # Corner patch size: 7% of the smaller image dimension, minimum 20px
    patch = max(20, int(min(h, w) * 0.07))

    # Convert to LAB for perceptually uniform distance computation
    img_uint8 = (image * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sample all 4 corner patches
    bg_pixels_lab = np.vstack([
        img_lab[:patch, :patch].reshape(-1, 3),           # top-left
        img_lab[:patch, w - patch:].reshape(-1, 3),       # top-right
        img_lab[h - patch:, :patch].reshape(-1, 3),       # bottom-left
        img_lab[h - patch:, w - patch:].reshape(-1, 3),   # bottom-right
    ])

    # Background model: mean and std per LAB channel
    bg_mean = np.mean(bg_pixels_lab, axis=0)
    bg_std = np.std(bg_pixels_lab, axis=0)
    # Floor std to avoid over-sensitivity on perfectly uniform backgrounds
    bg_std = np.maximum(bg_std, 5.0)

    logger.debug(
        f"BG color model (LAB): mean={bg_mean.round(1)}, std={bg_std.round(1)}"
    )

    # Per-pixel Mahalanobis-like distance from background distribution
    diff = (img_lab - bg_mean) / bg_std
    dist_map = np.sqrt(np.sum(diff ** 2, axis=2))

    # Foreground: pixels more than 3 sigma away from background mean
    fg_mask = (dist_map > 3.0).astype(np.uint8) * 255

    fg_ratio = np.count_nonzero(fg_mask) / image_area
    logger.debug(f"BG color: foreground coverage = {fg_ratio:.3f}")

    return _mask_to_quad(fg_mask, h, w, image_area, min_area_ratio, max_area_ratio)


def _find_boundary_via_grabcut(
    image: np.ndarray,
    min_area_ratio: float,
    max_area_ratio: float = _MASK_MAX_AREA_RATIO,
    iterations: int = 5,
    max_dimension: int = 800,
) -> Optional[np.ndarray]:
    """Detect subject boundary using GrabCut iterative segmentation.

    Initializes GrabCut with a rectangle inset 2% from the image edges.
    GrabCut iteratively refines foreground/background using Gaussian Mixture
    Models. Most robust for prints on varied backgrounds.

    Runs on a downsampled copy (longest side capped at max_dimension) to keep
    runtime practical on high-resolution inputs (3000×4000 → 600×800 ≈ 25×
    fewer pixels). Corner coordinates are scaled back to the original resolution.

    Args:
        image: Input image as float32 RGB [0, 1].
        min_area_ratio: Minimum valid subject area as fraction of total image area.
        max_area_ratio: Maximum valid subject area as fraction of total image area.
        iterations: Number of GrabCut EM iterations (default 5).
        max_dimension: Longest side of the downsampled image fed to GrabCut.

    Returns:
        Ordered corner points as (4, 2) float32 array in original image coordinates,
        or None if not found.
    """
    h, w = image.shape[:2]

    # Downsample before GrabCut — we only need coarse segmentation, so low
    # resolution is sufficient and dramatically faster (time scales with pixels).
    longest = max(h, w)
    if longest > max_dimension:
        scale = max_dimension / longest
        gc_h = max(50, int(h * scale))
        gc_w = max(50, int(w * scale))
        gc_uint8 = cv2.resize(
            (image * 255).astype(np.uint8),
            (gc_w, gc_h),
            interpolation=cv2.INTER_AREA,
        )
        scale_x = w / gc_w
        scale_y = h / gc_h
    else:
        gc_h, gc_w = h, w
        gc_uint8 = (image * 255).astype(np.uint8)
        scale_x = scale_y = 1.0

    gc_bgr = cv2.cvtColor(gc_uint8, cv2.COLOR_RGB2BGR)

    # Init rect: 2% inset from each edge. GrabCut treats everything inside
    # the rect as "probably foreground" for the first iteration.
    margin_x = max(2, int(gc_w * 0.02))
    margin_y = max(2, int(gc_h * 0.02))
    rect = (margin_x, margin_y, gc_w - 2 * margin_x, gc_h - 2 * margin_y)

    gc_mask = np.zeros((gc_h, gc_w), dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            gc_bgr, gc_mask, rect,
            bg_model, fg_model,
            iterations, cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error as e:
        logger.warning(f"GrabCut failed: {e}")
        return None

    # Definite foreground (GC_FGD) + probable foreground (GC_PR_FGD)
    fg_mask = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        np.uint8(255),
        np.uint8(0),
    )

    fg_ratio = np.count_nonzero(fg_mask) / (gc_h * gc_w)
    logger.debug(
        f"GrabCut: foreground coverage = {fg_ratio:.3f} "
        f"(ran on {gc_w}×{gc_h}, scale={1/scale_x:.2f}×)"
    )

    corners_small = _mask_to_quad(
        fg_mask, gc_h, gc_w, float(gc_h * gc_w), min_area_ratio, max_area_ratio,
    )
    if corners_small is None:
        return None

    # Scale corners back to original image dimensions
    corners_full = corners_small.copy()
    corners_full[:, 0] = np.clip(corners_full[:, 0] * scale_x, 0, w - 1)
    corners_full[:, 1] = np.clip(corners_full[:, 1] * scale_y, 0, h - 1)
    return corners_full


def detect_page(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area_ratio: float = 0.3,
    max_area_ratio: float = _EDGE_MAX_AREA_RATIO,
    method: str = "auto",
) -> PageDetection:
    """Detect album page or photo print boundaries in an image.

    Args:
        image: Input image as float32 RGB [0, 1], shape (H, W, 3).
        blur_kernel: Gaussian blur kernel size for Canny edge detection.
        canny_low: Lower threshold for Canny edge detector.
        canny_high: Upper threshold for Canny edge detector.
        min_area_ratio: Minimum detected area as fraction of total image area.
        max_area_ratio: Maximum area ratio for edge-based detection. Contours above
            this are rejected as background-perimeter false positives. Only applies
            to the "edge" method. Default 0.95.
        method: Detection algorithm — "edge", "color", "grabcut", or "auto".
            "auto" tries edge → color → grabcut in sequence.

    Returns:
        PageDetection with detected corners, confidence, and full-frame flag.
    """
    if method not in ("edge", "color", "grabcut", "auto"):
        raise ValueError(
            f"Unknown page detection method: {method!r}. "
            "Use 'edge', 'color', 'grabcut', or 'auto'."
        )

    height, width = image.shape[:2]
    image_area = float(height * width)

    corners: Optional[np.ndarray] = None
    method_used = "none"

    # -------------------------------------------------------------------------
    # Edge-based detection (4 internal strategies, all using max_area_ratio)
    # -------------------------------------------------------------------------
    def _try_edge() -> Optional[np.ndarray]:
        nonlocal method_used

        img_uint8 = (image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Strategy 1: normal Canny → contour
        edges = cv2.Canny(blurred, canny_low, canny_high)
        found = _find_largest_quadrilateral(
            cv2.dilate(edges, kernel3, iterations=1),
            image_area, min_area_ratio, max_area_ratio,
        )
        if found is not None:
            method_used = "edge_contour"
            return found

        # Strategy 2: stronger Canny (lower thresholds) → contour
        edges_strong = cv2.Canny(blurred, canny_low // 2, canny_high // 2)
        found = _find_largest_quadrilateral(
            cv2.dilate(edges_strong, kernel3, iterations=2),
            image_area, min_area_ratio, max_area_ratio,
        )
        if found is not None:
            method_used = "edge_contour_strong"
            return found

        # Strategy 3: Hough lines
        found = _find_quad_via_hough(
            edges, (height, width), min_area_ratio, max_area_ratio,
        )
        if found is not None:
            method_used = "edge_hough"
            return found

        # Strategy 4: adaptive threshold → contour
        adaptive = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            21, 5,
        )
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel3, iterations=3)
        found = _find_largest_quadrilateral(
            adaptive, image_area, min_area_ratio, max_area_ratio,
        )
        if found is not None:
            method_used = "edge_adaptive"
            return found

        return None

    # -------------------------------------------------------------------------
    # Dispatch
    # -------------------------------------------------------------------------
    if method == "edge":
        corners = _try_edge()

    elif method == "color":
        corners = _find_boundary_via_bg_color(image, min_area_ratio)
        if corners is not None:
            method_used = "color"

    elif method == "grabcut":
        corners = _find_boundary_via_grabcut(image, min_area_ratio)
        if corners is not None:
            method_used = "grabcut"

    elif method == "auto":
        # Fastest first
        corners = _try_edge()

        if corners is None:
            logger.debug("Edge detection found nothing; trying background color segmentation")
            corners = _find_boundary_via_bg_color(image, min_area_ratio)
            if corners is not None:
                method_used = "color"

        if corners is None:
            logger.debug("Color segmentation found nothing; trying GrabCut")
            corners = _find_boundary_via_grabcut(image, min_area_ratio)
            if corners is not None:
                method_used = "grabcut"

    # -------------------------------------------------------------------------
    # Full-frame fallback
    # -------------------------------------------------------------------------
    if corners is None:
        corners = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ], dtype=np.float32)
        logger.info("No page boundary detected, using full frame")
        return PageDetection(
            corners=corners,
            confidence=0.0,
            is_full_frame=True,
            method_used="full_frame",
            area_ratio=1.0,
        )

    # -------------------------------------------------------------------------
    # Confidence
    # -------------------------------------------------------------------------
    quad_area = cv2.contourArea(corners.astype(np.int32))
    area_ratio = quad_area / image_area

    _, (br_w, br_h), _ = cv2.minAreaRect(corners.astype(np.int32))
    rectangularity = quad_area / max(br_w * br_h, 1.0)

    confidence = min(1.0, area_ratio * rectangularity * 2.0)

    logger.info(
        f"Page detected via {method_used}: area_ratio={area_ratio:.3f}, "
        f"rectangularity={rectangularity:.3f}, confidence={confidence:.3f}"
    )

    return PageDetection(
        corners=corners,
        confidence=confidence,
        is_full_frame=False,
        method_used=method_used,
        area_ratio=area_ratio,
    )


def draw_page_detection(
    image: np.ndarray,
    detection: PageDetection,
) -> np.ndarray:
    """Draw page detection overlay on image for debugging.

    Args:
        image: Input image as float32 RGB [0, 1].
        detection: PageDetection result.

    Returns:
        Image with overlay as float32 RGB [0, 1].
    """
    img_uint8 = (image * 255).astype(np.uint8).copy()
    corners = detection.corners.astype(np.int32)

    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(img_uint8, pt1, pt2, (0, 255, 0), 3)

    for corner in corners:
        cv2.circle(img_uint8, tuple(corner), 10, (255, 0, 0), -1)

    text = f"conf:{detection.confidence:.2f} method:{detection.method_used}"
    if detection.is_full_frame:
        text += " (full_frame)"
    cv2.putText(img_uint8, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    return img_uint8.astype(np.float32) / 255.0
