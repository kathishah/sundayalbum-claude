"""Photo detection — identify individual photos on album pages."""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PhotoDetection:
    """Information about a detected photo."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    corners: np.ndarray  # 4x2 array of corner points (clockwise from top-left)
    confidence: float  # 0.0 to 1.0
    orientation: str  # "portrait", "landscape", or "square"
    area_ratio: float  # Ratio of photo area to page area
    contour: np.ndarray  # Original contour from detection


def detect_photos(
    page_image: np.ndarray,
    min_area_ratio: float = 0.05,
    max_count: int = 8,
    method: str = "contour",
) -> List[PhotoDetection]:
    """Detect individual photos on an album page or single print.

    Critical test cases:
    - IMG_three_pics: album page with 3 photos → should return 3 detections
    - IMG_two_pics_vertical_horizontal: 2 photos (portrait + landscape) → should return 2
    - IMG_cave, IMG_harbor, IMG_skydiving: single prints → should return 1 each

    Args:
        page_image: Album page image, float32 RGB [0, 1]
        min_area_ratio: Minimum photo area as ratio of total image area
        max_count: Maximum number of photos to detect
        method: Detection method ("contour", "claude", or "auto")

    Returns:
        List of PhotoDetection objects, sorted top-to-bottom, left-to-right
    """
    if method == "contour" or method == "auto":
        detections = _detect_photos_contour(page_image, min_area_ratio, max_count)
    elif method == "claude":
        raise NotImplementedError("Claude Vision API fallback not yet implemented")
    else:
        raise ValueError(f"Unknown detection method: {method}")

    logger.info(f"Detected {len(detections)} photos using method '{method}'")

    return detections


def _detect_photos_contour(
    page_image: np.ndarray,
    min_area_ratio: float,
    max_count: int,
) -> List[PhotoDetection]:
    """Detect photos using contour-based approach.

    The algorithm:
    1. Convert to grayscale
    2. Apply adaptive thresholding to separate photos from album page background
    3. Find contours
    4. Filter by area, aspect ratio, and shape
    5. Approximate to quadrilaterals
    6. Sort and return

    Args:
        page_image: Album page image, float32 RGB [0, 1]
        min_area_ratio: Minimum area ratio
        max_count: Maximum detections

    Returns:
        List of PhotoDetection objects
    """
    h, w = page_image.shape[:2]
    total_area = h * w
    min_area = total_area * min_area_ratio

    # Convert to uint8 for OpenCV
    image_uint8 = (page_image * 255).astype(np.uint8)

    # Convert to grayscale
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding works well for album pages where local contrast varies
    # Use a large block size to capture photo-level boundaries, not texture details
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Gaussian weights for smoother results
        cv2.THRESH_BINARY_INV,  # Invert: photos are often darker than album background
        blockSize=101,  # Large block to ignore local details
        C=15,  # Constant to subtract
    )

    # Morphological operations to clean up and connect photo regions
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If we found too few large contours, try the inverted threshold
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]

    if len(large_contours) == 0:
        # Try normal threshold (photos lighter than background)
        binary_alt = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,  # Not inverted
            blockSize=101,
            C=15,
        )
        binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_CLOSE, kernel_close)
        binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_OPEN, kernel_open)

        contours_alt, _ = cv2.findContours(binary_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_contours_alt = [c for c in contours_alt if cv2.contourArea(c) > min_area]

        if len(large_contours_alt) > 0:
            contours = contours_alt
            binary = binary_alt
            large_contours = large_contours_alt
            logger.debug(f"Using non-inverted threshold ({len(large_contours_alt)} large contours)")
        else:
            logger.debug(f"Both thresholds found 0 large contours - will treat as single photo")
    else:
        logger.debug(f"Using inverted threshold ({len(large_contours)} large contours)")

    logger.debug(f"Found {len(contours)} contours before area/shape filtering")

    # Filter and classify contours
    detections: List[PhotoDetection] = []
    filtered_reasons = []

    for idx, contour in enumerate(contours):
        # Filter by area
        area = cv2.contourArea(contour)
        if area < min_area:
            area_ratio = area / total_area
            filtered_reasons.append(f"Contour {idx}: area too small ({area:.0f} px, {area_ratio:.3f} ratio)")
            continue

        # Filter by shape — photos should be roughly rectangular
        # Compute perimeter and check if the contour is convex-ish
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            filtered_reasons.append(f"Contour {idx}: zero perimeter")
            continue

        # Approximate to polygon
        epsilon = 0.02 * perimeter  # Approximation accuracy
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # We expect 4-sided shapes (quadrilaterals) for photos
        # But be flexible — 4 to 12 sides is OK (relaxed from 8)
        num_vertices = len(approx)
        if num_vertices < 4 or num_vertices > 12:
            filtered_reasons.append(f"Contour {idx}: bad vertex count ({num_vertices})")
            continue

        # Get bounding box
        x, y, bw, bh = cv2.boundingRect(contour)

        # Filter by aspect ratio — photos are typically 3:2, 4:3, 1:1, etc.
        # Reject extremely elongated shapes (likely not photos)
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect_ratio > 6.0:  # Too elongated (relaxed from 5.0)
            area_ratio_check = area / total_area
            filtered_reasons.append(
                f"Contour {idx}: aspect ratio too high ({aspect_ratio:.2f}, area={area:.0f}, ratio={area_ratio_check:.3f})"
            )
            continue

        # Compute area ratio relative to page
        area_ratio = area / total_area

        # If we have more than 4 vertices, find best-fit quadrilateral
        if num_vertices > 4:
            # Use minimum area rectangle as approximation
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = np.array(box, dtype=np.float32)
        else:
            corners = approx.reshape(-1, 2).astype(np.float32)

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = _order_corners(corners)

        # Determine orientation based on aspect ratio of bounding box
        if bw > bh * 1.1:
            orientation = "landscape"
        elif bh > bw * 1.1:
            orientation = "portrait"
        else:
            orientation = "square"

        # Compute confidence based on:
        # 1. How rectangular the shape is (ratio of area to bounding box area)
        # 2. How clean the approximation is (fewer vertices = cleaner)
        rect_area = bw * bh
        rectangularity = area / (rect_area + 1e-6)
        vertex_score = 1.0 - (abs(num_vertices - 4) / 10.0)  # 4 vertices is ideal
        confidence = (rectangularity + vertex_score) / 2.0
        confidence = np.clip(confidence, 0.0, 1.0)

        detection = PhotoDetection(
            bbox=(x, y, x + bw, y + bh),
            corners=corners,
            confidence=confidence,
            orientation=orientation,
            area_ratio=area_ratio,
            contour=contour,
        )

        detections.append(detection)

    # Sort detections by confidence (descending)
    detections.sort(key=lambda d: d.confidence, reverse=True)

    # Limit to max_count
    detections = detections[:max_count]

    # Re-sort by position (top-to-bottom, left-to-right) for user-friendly ordering
    detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))

    logger.debug(
        f"After filtering: {len(detections)} detections "
        f"(min_area={min_area:.0f}px, min_ratio={min_area_ratio:.3f})"
    )

    # Log filtering reasons for debugging
    if filtered_reasons:
        logger.debug(f"Filtered out {len(filtered_reasons)} contours:")
        for reason in filtered_reasons[:10]:  # Limit to first 10
            logger.debug(f"  {reason}")

    return detections


def _order_corners(corners: np.ndarray) -> np.ndarray:
    """Order corners in clockwise order starting from top-left.

    Args:
        corners: Nx2 array of corner points (N should be 4)

    Returns:
        4x2 array ordered as: top-left, top-right, bottom-right, bottom-left
    """
    # If we have exactly 4 corners, use standard ordering
    if len(corners) == 4:
        # Sum and difference of coordinates to find corners
        # Top-left has smallest sum (x+y)
        # Bottom-right has largest sum
        # Top-right has smallest difference (y-x)
        # Bottom-left has largest difference
        sums = corners.sum(axis=1)
        diffs = np.diff(corners, axis=1).flatten()

        top_left = corners[np.argmin(sums)]
        bottom_right = corners[np.argmax(sums)]
        top_right = corners[np.argmin(diffs)]
        bottom_left = corners[np.argmax(diffs)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    else:
        # If not exactly 4 corners, just return as-is
        # (This shouldn't happen if filtering worked correctly)
        logger.warning(f"Expected 4 corners, got {len(corners)}")
        return corners


def draw_photo_detections(
    page_image: np.ndarray,
    detections: List[PhotoDetection],
) -> np.ndarray:
    """Draw detected photo boundaries on the page image for visualization.

    Args:
        page_image: Album page image, float32 RGB [0, 1]
        detections: List of photo detections

    Returns:
        Visualization image with colored boxes and labels, float32 RGB [0, 1]
    """
    # Convert to uint8 for drawing
    vis = (page_image * 255).astype(np.uint8).copy()

    # Color palette for multiple photos (BGR for OpenCV)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 255, 0),  # Spring green
        (255, 128, 0),  # Orange
    ]

    for i, detection in enumerate(detections):
        color = colors[i % len(colors)]

        # Draw corners as polygon
        corners_int = detection.corners.astype(np.int32)
        cv2.polylines(vis, [corners_int], isClosed=True, color=color, thickness=3)

        # Draw corner dots
        for corner in corners_int:
            cv2.circle(vis, tuple(corner), 8, (0, 0, 255), -1)  # Red dots

        # Draw bounding box (lighter, dashed-like)
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=1)

        # Label with photo number and confidence
        label = f"Photo {i+1} ({detection.confidence:.2f}) {detection.orientation}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Draw text background
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        cv2.rectangle(
            vis,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            color,
            -1  # Filled
        )

        # Draw text
        cv2.putText(
            vis,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            thickness,
        )

    # Convert back to float32
    return vis.astype(np.float32) / 255.0
