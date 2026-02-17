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
    min_area_ratio: float = 0.02,  # Lowered from 0.05 to 0.02 (2%) to handle album pages with visible background
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

    # Morphological operations to clean up photo regions
    # Balance: kernels must be large enough to connect fragmented photos,
    # but small enough to NOT merge separate photos together
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))  # Connect fragments
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))  # Clean noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # Find contours - use RETR_LIST to get ALL contours, not just external ones
    # This is critical for album pages where individual photos are INSIDE the page boundary
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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
        # Use same balanced morphological operations
        kernel_close_alt = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        kernel_open_alt = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_CLOSE, kernel_close_alt)
        binary_alt = cv2.morphologyEx(binary_alt, cv2.MORPH_OPEN, kernel_open_alt)

        contours_alt, _ = cv2.findContours(binary_alt, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Only count contours that survive the second-pass area filter (< 98% of image).
        # A nearly-full-frame contour (e.g. 99.8%) means the threshold made the entire
        # image white and is not a useful detection; fall through to Canny in that case.
        large_contours_alt = [
            c for c in contours_alt
            if min_area < cv2.contourArea(c) <= total_area * 0.98
        ]

        if len(large_contours_alt) > 0:
            contours = contours_alt
            binary = binary_alt
            large_contours = large_contours_alt
            logger.debug(f"Using non-inverted threshold ({len(large_contours_alt)} large contours)")
        else:
            # Both adaptive thresholds failed - try Canny edge detection as last resort
            logger.debug(f"Both thresholds found 0 large contours - trying Canny edge detection fallback")

            # Canny edge detection with moderate thresholds
            edges = cv2.Canny(blurred, 50, 150)

            # Dilate edges to connect broken boundaries
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            edges_dilated = cv2.dilate(edges, kernel_dilate, iterations=2)

            # Fill holes to create solid regions
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            binary_canny = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel_close)

            contours_canny, _ = cv2.findContours(binary_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            large_contours_canny = [c for c in contours_canny if cv2.contourArea(c) > min_area]

            if len(large_contours_canny) > 0:
                contours = contours_canny
                binary = binary_canny
                large_contours = large_contours_canny
                logger.debug(f"Canny fallback successful: {len(large_contours_canny)} large contours")
            else:
                logger.debug(f"Canny fallback also found 0 large contours - will return empty list")
    else:
        logger.debug(f"Using inverted threshold ({len(large_contours)} large contours)")

    logger.debug(f"Found {len(contours)} contours before area/shape filtering")

    # Filter and classify contours
    detections: List[PhotoDetection] = []
    filtered_reasons = []

    # First pass: collect all potentially valid contours
    # We'll apply different max_area thresholds based on whether this looks like a single print or album page
    potential_detections = []

    for idx, contour in enumerate(contours):
        # Filter by area
        area = cv2.contourArea(contour)
        area_ratio = area / total_area

        # Too small: likely noise/decoration
        if area < min_area:
            filtered_reasons.append(f"Contour {idx}: area too small ({area:.0f} px, {area_ratio:.3f} ratio)")
            continue

        # Skip extremely large contours (>98%) - these are definitely full-frame/page boundaries
        # Note: Single prints can be 95-98% of image, so we use 0.98 here and let adaptive logic handle it
        if area_ratio > 0.98:
            filtered_reasons.append(f"Contour {idx}: area too large ({area:.0f} px, {area_ratio:.3f} ratio) - likely full frame")
            continue

        # Store for later analysis
        potential_detections.append((idx, contour, area, area_ratio))

    # Determine if this is a single print or multi-photo page
    # Single print scenario: 1-2 large contours (30-95% of image)
    # Multi-photo page: multiple smaller contours (each 2-30% of image)
    large_contours = [d for d in potential_detections if d[3] >= 0.30]  # >= 30% area
    medium_contours = [d for d in potential_detections if 0.10 <= d[3] < 0.30]

    is_single_print = (len(large_contours) == 1 and len(medium_contours) == 0)

    logger.debug(
        f"Detection scenario analysis: large_contours={len(large_contours)}, "
        f"medium_contours={len(medium_contours)}, "
        f"total_potential={len(potential_detections)}, "
        f"is_single_print={is_single_print}"
    )

    # Apply appropriate max_area threshold based on scenario
    if is_single_print:
        # Single print: allow contours up to 98% (we filtered >98% in first pass)
        # Single prints can be very close to image edges, so we're generous here
        max_area_threshold = 0.98
        logger.debug("Using single-print mode: max_area_threshold=0.98")
    else:
        # Multi-photo page: individual photos typically < 40% of total image
        # Page boundary would be >90%, so use 0.90 as max
        max_area_threshold = 0.90
        logger.debug("Using multi-photo mode: max_area_threshold=0.90")

    # Second pass: apply max_area filter and shape analysis
    for idx, contour, area, area_ratio in potential_detections:
        # Apply max_area threshold
        if area_ratio > max_area_threshold:
            filtered_reasons.append(
                f"Contour {idx}: area too large ({area:.0f} px, {area_ratio:.3f} ratio) - "
                f"exceeds threshold {max_area_threshold}"
            )
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
            filtered_reasons.append(
                f"Contour {idx}: aspect ratio too high ({aspect_ratio:.2f}, area={area:.0f}, ratio={area_ratio:.3f})"
            )
            continue

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

    # Filter out small decorations that are significantly smaller than real photos
    # Use adaptive threshold based on number of detections
    if len(detections) > 1:
        areas = [d.area_ratio * total_area for d in detections]
        max_area = max(areas)

        # Adaptive threshold for decoration filtering:
        # Album pages can have photos of varying sizes (e.g., 3×5 mixed with 4×6)
        # Use relaxed threshold to avoid filtering out legitimately smaller photos
        # - If we have many detections (> 4), use moderate filter (30%)
        # - If we have few detections (2-4), use very loose filter (20%)
        threshold = 0.30 if len(detections) > 4 else 0.20

        logger.debug(
            f"Decoration filter check: {len(detections)} detections, "
            f"max_area={max_area:.0f}px, threshold={threshold*100:.0f}%"
        )

        filtered_detections = []
        for i, (detection, area) in enumerate(zip(detections, areas), 1):
            ratio = area / max_area
            if ratio >= threshold:
                filtered_detections.append(detection)
                logger.debug(f"  Detection {i}: {area:.0f}px ({ratio*100:.1f}%) -> KEEP")
            else:
                logger.debug(f"  Detection {i}: {area:.0f}px ({ratio*100:.1f}%) -> FILTER")

        logger.debug(f"After decoration filter: {len(filtered_detections)} remaining")
        detections = filtered_detections

    # Remove overlapping detections (keep higher confidence / larger ones)
    detections = _remove_overlapping_detections(detections, iou_threshold=0.3)

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

    # Sanity check: if 2+ detections have badly disproportionate areas (one is >2.5x
    # another), the contour method likely found wrong boundaries.  Fall back to the
    # projection-profile approach which finds physical gaps between photos.
    if len(detections) >= 2:
        areas = [d.area_ratio for d in detections]
        area_imbalance = max(areas) / (min(areas) + 1e-6)
        if area_imbalance > 2.5:
            logger.warning(
                f"Contour detections are disproportionate (imbalance={area_imbalance:.1f}x). "
                f"Falling back to projection-profile detection."
            )
            proj_detections = _detect_photos_by_projection(page_image, min_area_ratio)
            if proj_detections:
                detections = proj_detections

    return detections


def _detect_photos_by_projection(
    page_image: np.ndarray,
    min_area_ratio: float,
) -> List[PhotoDetection]:
    """Detect photo split lines using edge-density projection profiles.

    For pages where contour detection produces badly disproportionate regions this
    approach is more robust.  It computes Sobel-edge magnitude sums across every
    row and column, then locates the deepest valley — the physical gap (album page
    divider) between adjacent photos — and uses it as the split line.

    Currently supports pages with exactly 2 photos separated by one dominant gap
    (vertical side-by-side or horizontal stacked layout).

    Args:
        page_image: Album page image, float32 RGB [0, 1].
        min_area_ratio: Minimum photo area as fraction of total image area.

    Returns:
        List of two PhotoDetection objects, or empty list if no clear gap found.
    """
    h, w = page_image.shape[:2]
    total_area = float(h * w)

    image_uint8 = (page_image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Sobel edge magnitude captures structural boundaries between photos
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Smooth projection profiles to find broad valleys (album dividers are wide)
    smooth_k = max(11, min(h, w) // 30)

    row_profile = edge_mag.sum(axis=1)  # (h,)  — rows with few edges = horizontal gap
    col_profile = edge_mag.sum(axis=0)  # (w,)  — cols with few edges = vertical gap

    row_smooth = np.convolve(row_profile, np.ones(smooth_k) / smooth_k, mode='same')
    col_smooth = np.convolve(col_profile, np.ones(smooth_k) / smooth_k, mode='same')

    # Search for the valley only in the central 20–80% of each axis to avoid borders
    margin_h = int(h * 0.20)
    margin_w = int(w * 0.20)

    h_search = row_smooth[margin_h: h - margin_h]
    v_search = col_smooth[margin_w: w - margin_w]

    h_min_idx = int(np.argmin(h_search)) + margin_h
    v_min_idx = int(np.argmin(v_search)) + margin_w

    # Relative valley depth: lower ratio = cleaner / more definitive gap
    mean_row = row_smooth.mean() + 1e-6
    mean_col = col_smooth.mean() + 1e-6
    h_relative = row_smooth[h_min_idx] / mean_row
    v_relative = col_smooth[v_min_idx] / mean_col

    logger.debug(
        f"Projection valleys — horizontal at y={h_min_idx} (rel={h_relative:.3f}), "
        f"vertical at x={v_min_idx} (rel={v_relative:.3f})"
    )

    # Choose the axis with the deeper (lower-relative) valley
    if v_relative <= h_relative:
        split_x = v_min_idx
        regions: List[Tuple[int, int, int, int]] = [
            (0, 0, split_x, h),
            (split_x, 0, w, h),
        ]
        logger.debug(f"Projection: vertical split at x={split_x}")
    else:
        split_y = h_min_idx
        regions = [
            (0, 0, w, split_y),
            (0, split_y, w, h),
        ]
        logger.debug(f"Projection: horizontal split at y={split_y}")

    detections: List[PhotoDetection] = []
    for x1, y1, x2, y2 in regions:
        bw, bh = x2 - x1, y2 - y1
        area_ratio = (bw * bh) / total_area

        if area_ratio < min_area_ratio:
            continue

        corners = np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32
        )

        if bw > bh * 1.1:
            orientation = "landscape"
        elif bh > bw * 1.1:
            orientation = "portrait"
        else:
            orientation = "square"

        contour = np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32
        )

        detections.append(PhotoDetection(
            bbox=(x1, y1, x2, y2),
            corners=corners,
            confidence=0.75,
            orientation=orientation,
            area_ratio=area_ratio,
            contour=contour,
        ))

    logger.debug(f"Projection detection produced {len(detections)} regions")
    return detections


def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: First bbox as (x1, y1, x2, y2)
        bbox2: Second bbox as (x1, y1, x2, y2)

    Returns:
        IoU value between 0.0 and 1.0
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0  # No overlap

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _remove_overlapping_detections(
    detections: List[PhotoDetection],
    iou_threshold: float = 0.3
) -> List[PhotoDetection]:
    """Remove overlapping detections using Non-Maximum Suppression.

    Keep detections with higher confidence when IoU > threshold.

    Args:
        detections: List of detections (should be sorted by confidence desc)
        iou_threshold: IoU threshold for considering detections as overlapping

    Returns:
        Filtered list with no significant overlaps
    """
    if len(detections) <= 1:
        return detections

    # Sort by confidence (descending) to prioritize keeping high-confidence detections
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)

    keep = []
    for i, det1 in enumerate(sorted_dets):
        # Check if det1 overlaps significantly with any already-kept detection
        overlaps = False
        for det2 in keep:
            iou = _compute_iou(det1.bbox, det2.bbox)
            if iou > iou_threshold:
                overlaps = True
                logger.debug(f"Removing overlapping detection (IoU={iou:.2f}): "
                           f"area={det1.area_ratio:.2%}, conf={det1.confidence:.2f}")
                break

        if not overlaps:
            keep.append(det1)

    logger.debug(f"Non-maximum suppression: {len(detections)} -> {len(keep)} detections")
    return keep


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
