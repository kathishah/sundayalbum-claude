"""Detect album page boundaries in images.

Handles two categories of input:
- Album pages (multi-photo): detects the album page boundary
- Individual prints: detects the photo print edges
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PageDetection:
    """Result of page boundary detection."""

    corners: np.ndarray  # shape (4, 2) — four corner points as (x, y)
    confidence: float  # 0.0 to 1.0
    is_full_frame: bool  # True if no clear boundary found, using full image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four corner points as: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: Array of shape (4, 2) with (x, y) coordinates.

    Returns:
        Ordered array of shape (4, 2).
    """
    # Sort by sum (x+y): smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    # Sort by diff (y-x): smallest = top-right, largest = bottom-left
    d = np.diff(pts, axis=1).flatten()

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left

    return ordered


def _find_largest_quadrilateral(
    edges: np.ndarray,
    image_area: float,
    min_area_ratio: float,
) -> Optional[np.ndarray]:
    """Find the largest quadrilateral contour in an edge image.

    Args:
        edges: Binary edge image from Canny.
        image_area: Total image area (height * width).
        min_area_ratio: Minimum contour area as fraction of image area.

    Returns:
        Ordered corner points as (4, 2) array, or None if not found.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Sort by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip contours that are too small
        if area < min_area_ratio * image_area:
            continue

        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # We want a quadrilateral (4 vertices)
        if len(approx) == 4:
            corners = approx.reshape(4, 2).astype(np.float32)
            return _order_corners(corners)

    return None


def _find_quad_via_hough(
    edges: np.ndarray,
    image_shape: tuple,
    min_area_ratio: float,
) -> Optional[np.ndarray]:
    """Find page quadrilateral using Hough line detection.

    Fallback method when contour-based detection doesn't find a clean quad.

    Args:
        edges: Binary edge image.
        image_shape: (height, width) of the image.
        min_area_ratio: Minimum area ratio for valid detection.

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

    # Classify lines as roughly horizontal or roughly vertical
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

    # Find the top/bottom horizontal lines and left/right vertical lines
    h_lines = np.array(horizontal_lines)
    v_lines = np.array(vertical_lines)

    # Sort horizontal lines by y-midpoint
    h_y_mids = (h_lines[:, 1] + h_lines[:, 3]) / 2
    h_sorted_idx = np.argsort(h_y_mids)
    top_line = h_lines[h_sorted_idx[0]]
    bottom_line = h_lines[h_sorted_idx[-1]]

    # Sort vertical lines by x-midpoint
    v_x_mids = (v_lines[:, 0] + v_lines[:, 2]) / 2
    v_sorted_idx = np.argsort(v_x_mids)
    left_line = v_lines[v_sorted_idx[0]]
    right_line = v_lines[v_sorted_idx[-1]]

    # Find intersections of the 4 boundary lines
    def line_intersection(l1: np.ndarray, l2: np.ndarray) -> Optional[np.ndarray]:
        """Find intersection of two line segments."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return np.array([ix, iy], dtype=np.float32)

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

    corners = np.array(corners, dtype=np.float32)

    # Validate: corners should be within image bounds (with some margin)
    margin = max(height, width) * 0.1
    if np.any(corners < -margin) or np.any(corners[:, 0] > width + margin) or np.any(corners[:, 1] > height + margin):
        return None

    # Clip to image bounds
    corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)

    # Validate area
    ordered = _order_corners(corners)
    quad_area = cv2.contourArea(ordered.astype(np.int32))
    if quad_area < min_area_ratio * image_area:
        return None

    return ordered


def detect_page(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area_ratio: float = 0.3,
) -> PageDetection:
    """Detect album page or photo print boundaries in an image.

    Works on two categories:
    - Album pages (three_pics, two_pics): detects the album page boundary
    - Individual prints (cave, harbor, skydiving): detects print edges

    If no clear boundary is found, returns full image bounds with is_full_frame=True.

    Args:
        image: Input image as float32 RGB [0, 1], shape (H, W, 3).
        blur_kernel: Gaussian blur kernel size for edge detection.
        canny_low: Lower threshold for Canny edge detector.
        canny_high: Upper threshold for Canny edge detector.
        min_area_ratio: Minimum detected area as fraction of total image area.

    Returns:
        PageDetection with detected corners, confidence, and full-frame flag.
    """
    height, width = image.shape[:2]
    image_area = height * width

    # Convert to uint8 grayscale for edge detection
    img_uint8 = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Try multiple edge detection strategies for robustness
    corners = None
    method_used = "none"

    # Strategy 1: Canny edges → contour-based quadrilateral detection
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Dilate edges to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    corners = _find_largest_quadrilateral(edges_dilated, image_area, min_area_ratio)
    if corners is not None:
        method_used = "contour"

    # Strategy 2: If contour didn't work, try with stronger edge detection
    if corners is None:
        edges_strong = cv2.Canny(blurred, canny_low // 2, canny_high // 2)
        edges_strong = cv2.dilate(edges_strong, kernel, iterations=2)
        corners = _find_largest_quadrilateral(edges_strong, image_area, min_area_ratio)
        if corners is not None:
            method_used = "contour_strong"

    # Strategy 3: Hough lines fallback
    if corners is None:
        corners = _find_quad_via_hough(edges, image.shape, min_area_ratio)
        if corners is not None:
            method_used = "hough"

    # Strategy 4: Adaptive threshold + contour
    if corners is None:
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        # Close small gaps
        adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=3)
        corners = _find_largest_quadrilateral(adaptive, image_area, min_area_ratio)
        if corners is not None:
            method_used = "adaptive"

    # If no boundary found, use full image
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
        )

    # Compute confidence based on how well the quad covers the image
    # and how rectangular it is
    quad_area = cv2.contourArea(corners.astype(np.int32))
    area_ratio = quad_area / image_area

    # Check rectangularity: compare quad area to bounding rect area
    bounding_rect = cv2.minAreaRect(corners.astype(np.int32))
    _, (br_w, br_h), _ = bounding_rect
    bounding_area = br_w * br_h
    rectangularity = quad_area / max(bounding_area, 1.0)

    # Confidence: combine area ratio and rectangularity
    # A good detection covers a significant area and is roughly rectangular
    confidence = min(1.0, area_ratio * rectangularity * 2.0)

    logger.info(
        f"Page detected via {method_used}: area_ratio={area_ratio:.3f}, "
        f"rectangularity={rectangularity:.3f}, confidence={confidence:.3f}"
    )

    return PageDetection(
        corners=corners,
        confidence=confidence,
        is_full_frame=False,
    )


def draw_page_detection(
    image: np.ndarray,
    detection: PageDetection,
) -> np.ndarray:
    """Draw page detection overlay on image for debugging.

    Draws green quad boundary and red corner dots.

    Args:
        image: Input image as float32 RGB [0, 1].
        detection: PageDetection result.

    Returns:
        Image with overlay as float32 RGB [0, 1].
    """
    # Work in uint8 for drawing
    img_uint8 = (image * 255).astype(np.uint8).copy()

    corners = detection.corners.astype(np.int32)

    # Draw green quad outline
    for i in range(4):
        pt1 = tuple(corners[i])
        pt2 = tuple(corners[(i + 1) % 4])
        cv2.line(img_uint8, pt1, pt2, (0, 255, 0), 3)

    # Draw red corner dots
    for corner in corners:
        cv2.circle(img_uint8, tuple(corner), 10, (255, 0, 0), -1)

    # Add confidence text
    text = f"conf: {detection.confidence:.2f}"
    if detection.is_full_frame:
        text += " (full frame)"
    cv2.putText(
        img_uint8,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    return img_uint8.astype(np.float32) / 255.0
