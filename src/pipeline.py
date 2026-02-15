"""Main pipeline orchestrator for Sunday Album processing."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.preprocessing.loader import load_image, ImageMetadata
from src.preprocessing.normalizer import normalize, NormalizationResult
from src.page_detection.detector import detect_page, draw_page_detection, PageDetection
from src.page_detection.perspective import correct_perspective
from src.glare.detector import detect_glare, draw_glare_overlay, GlareDetection
from src.glare.confidence import compute_glare_confidence
from src.glare.remover_single import remove_glare_single, GlareResult
from src.photo_detection.detector import detect_photos, draw_photo_detections, PhotoDetection
from src.photo_detection.splitter import split_photos

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """All tunable parameters in one place."""

    # Preprocessing
    max_working_resolution: int = 4000  # px, longest edge
    prefer_heic_for_iteration: bool = True

    # Page detection
    page_detect_blur_kernel: int = 5
    page_detect_canny_low: int = 50
    page_detect_canny_high: int = 150
    page_detect_min_area_ratio: float = 0.3

    # Glare detection — two profiles for two glare types
    glare_intensity_threshold: float = 0.85
    glare_saturation_threshold: float = 0.15
    glare_min_area: int = 100
    glare_inpaint_radius: int = 5
    glare_feather_radius: int = 5
    glare_type: str = "auto"  # "auto", "sleeve" (flat plastic), or "print" (curved glossy)

    # Photo detection
    photo_detect_method: str = "contour"  # "contour", "yolo", or "claude"
    photo_detect_min_area_ratio: float = 0.02  # 2% of page area (was 0.05)
    photo_detect_max_count: int = 8

    # Geometry
    keystone_max_angle: float = 40.0
    rotation_auto_correct_max: float = 15.0

    # Color
    clahe_clip_limit: float = 2.0
    clahe_grid_size: tuple = (8, 8)
    sharpen_radius: float = 1.5
    sharpen_amount: float = 0.5
    saturation_boost: float = 0.15

    # Output
    output_format: str = "jpeg"
    jpeg_quality: int = 92

    # AI
    use_ai_quality_check: bool = False
    use_ai_fallback_detection: bool = False
    anthropic_model: str = "claude-sonnet-4-5-20250929"


@dataclass
class PipelineResult:
    """Result of pipeline processing."""

    output_images: List[np.ndarray]
    metadata: ImageMetadata
    processing_time: float
    steps_completed: List[str]
    page_detection: Optional[PageDetection] = None
    glare_detection: Optional[GlareDetection] = None
    glare_confidence: Optional[float] = None
    glare_removal: Optional[GlareResult] = None
    photo_detections: Optional[List[PhotoDetection]] = None
    num_photos_extracted: int = 0


class Pipeline:
    """Main processing pipeline for album page digitization."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration. If None, uses defaults.
        """
        self.config = config or PipelineConfig()
        self.step_times: dict = {}

    def process(
        self,
        input_path: str,
        debug_output_dir: Optional[str] = None,
        steps_filter: Optional[List[str]] = None
    ) -> PipelineResult:
        """Process a single image through the pipeline.

        Args:
            input_path: Path to input image
            debug_output_dir: Optional directory for debug output
            steps_filter: Optional list of step names to run. If None, runs all steps.

        Returns:
            PipelineResult with processed images and metadata
        """
        start_time = time.time()
        steps_completed: List[str] = []
        debug_dir: Optional[Path] = None

        if debug_output_dir:
            debug_dir = Path(debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing: {input_path}")

        # Step 1: Load image
        step_start = time.time()
        image, metadata = load_image(input_path)
        self.step_times['load'] = time.time() - step_start
        steps_completed.append('load')
        logger.info(f"Load time: {self.step_times['load']:.3f}s")

        if debug_dir:
            from src.utils.debug import save_debug_image
            save_debug_image(
                image,
                debug_dir / "01_loaded.jpg",
                "Loaded image with EXIF orientation applied"
            )

        # Step 2: Normalize (resize + thumbnail)
        step_start = time.time()
        norm_result = normalize(image, self.config.max_working_resolution)
        self.step_times['normalize'] = time.time() - step_start
        steps_completed.append('normalize')
        logger.info(f"Normalize time: {self.step_times['normalize']:.3f}s")

        working_image = norm_result.image

        # Step 3: Page detection & perspective correction
        page_detection_result: Optional[PageDetection] = None
        should_run_page_detect = steps_filter is None or 'page_detect' in steps_filter

        if should_run_page_detect:
            step_start = time.time()
            try:
                page_detection_result = detect_page(
                    working_image,
                    blur_kernel=self.config.page_detect_blur_kernel,
                    canny_low=self.config.page_detect_canny_low,
                    canny_high=self.config.page_detect_canny_high,
                    min_area_ratio=self.config.page_detect_min_area_ratio,
                )

                # Save debug: page boundary overlay
                if debug_dir:
                    from src.utils.debug import save_debug_image
                    overlay = draw_page_detection(working_image, page_detection_result)
                    save_debug_image(
                        overlay,
                        debug_dir / "02_page_detected.jpg",
                        f"Page detection (confidence={page_detection_result.confidence:.2f}, "
                        f"full_frame={page_detection_result.is_full_frame})"
                    )

                # Apply perspective correction if a boundary was found
                if not page_detection_result.is_full_frame:
                    working_image = correct_perspective(
                        working_image,
                        page_detection_result.corners,
                    )

                    if debug_dir:
                        from src.utils.debug import save_debug_image
                        save_debug_image(
                            working_image,
                            debug_dir / "03_page_warped.jpg",
                            "After perspective correction"
                        )
                else:
                    logger.info("Full frame detected, skipping perspective correction")

                self.step_times['page_detect'] = time.time() - step_start
                steps_completed.append('page_detect')
                logger.info(f"Page detection time: {self.step_times['page_detect']:.3f}s")

            except Exception as e:
                logger.warning(f"Page detection failed, passing through: {e}")
                self.step_times['page_detect'] = time.time() - step_start

        # Step 4: Photo detection & splitting (do this BEFORE glare removal)
        # Rationale: Better to detect individual photos first, then apply glare removal
        # to each photo separately. This gives better results than removing glare from
        # the whole page and then splitting.
        photo_detections_result: Optional[List[PhotoDetection]] = None
        extracted_photos: List[np.ndarray] = []
        should_run_photo_detect = steps_filter is None or 'photo_detect' in steps_filter

        if should_run_photo_detect:
            step_start = time.time()
            try:
                # Detect individual photos on the page
                photo_detections_result = detect_photos(
                    working_image,
                    min_area_ratio=self.config.photo_detect_min_area_ratio,
                    max_count=self.config.photo_detect_max_count,
                    method=self.config.photo_detect_method,
                )

                # Save debug: photo boundaries overlay
                if debug_dir:
                    from src.utils.debug import save_debug_image
                    overlay = draw_photo_detections(working_image, photo_detections_result)
                    save_debug_image(
                        overlay,
                        debug_dir / "04_photo_boundaries.jpg",
                        f"Detected {len(photo_detections_result)} photos"
                    )

                # Extract individual photos
                if photo_detections_result:
                    extracted_photos = split_photos(working_image, photo_detections_result)

                    # Save debug: each extracted photo (before glare removal)
                    if debug_dir:
                        from src.utils.debug import save_debug_image
                        for i, photo in enumerate(extracted_photos, 1):
                            save_debug_image(
                                photo,
                                debug_dir / f"05_photo_{i:02d}_raw.jpg",
                                f"Extracted photo {i} ({photo.shape[1]}x{photo.shape[0]})"
                            )

                self.step_times['photo_detect'] = time.time() - step_start
                steps_completed.append('photo_detect')
                logger.info(
                    f"Photo detection time: {self.step_times['photo_detect']:.3f}s, "
                    f"detected={len(photo_detections_result)}, extracted={len(extracted_photos)}"
                )

            except Exception as e:
                logger.warning(f"Photo detection failed, passing through: {e}")
                self.step_times['photo_detect'] = time.time() - step_start

        # If no photos were extracted, treat the whole page as one photo
        if not extracted_photos:
            extracted_photos = [working_image]
            logger.info("No photos detected, treating entire page as single photo")

        # Step 5: Glare detection & removal (per photo)
        # Now we process each extracted photo separately for glare
        glare_detection_result: Optional[GlareDetection] = None
        glare_confidence_score: Optional[float] = None
        should_run_glare_detect = steps_filter is None or 'glare' in steps_filter

        if should_run_glare_detect:
            step_start = time.time()
            deglared_photos: List[np.ndarray] = []

            try:
                # Process each photo separately for glare detection and removal
                for photo_idx, photo in enumerate(extracted_photos, 1):
                    logger.debug(f"Processing glare for photo {photo_idx}/{len(extracted_photos)}")

                    # Detect glare on this individual photo
                    photo_glare = detect_glare(
                        photo,
                        intensity_threshold=self.config.glare_intensity_threshold,
                        saturation_threshold=self.config.glare_saturation_threshold,
                        min_area=self.config.glare_min_area,
                        glare_type=self.config.glare_type,
                    )

                    # Save debug for first photo (to avoid clutter)
                    if debug_dir and photo_idx == 1:
                        from src.utils.debug import save_debug_image, save_debug_text

                        save_debug_image(
                            photo_glare.mask,
                            debug_dir / f"06_photo_{photo_idx:02d}_glare_mask.png",
                            f"Photo {photo_idx} glare mask"
                        )

                        overlay = draw_glare_overlay(photo, photo_glare)
                        save_debug_image(
                            overlay,
                            debug_dir / f"06_photo_{photo_idx:02d}_glare_overlay.jpg",
                            f"Photo {photo_idx} glare (type={photo_glare.glare_type})"
                        )

                    # Remove glare if detected
                    if photo_glare.total_glare_area_ratio > 0.001:
                        glare_result = remove_glare_single(
                            photo,
                            photo_glare.mask,
                            photo_glare.severity_map,
                            inpaint_radius=self.config.glare_inpaint_radius,
                            feather_radius=self.config.glare_feather_radius,
                        )
                        deglared_photo = glare_result.image

                        # Save debug
                        if debug_dir:
                            from src.utils.debug import save_debug_image
                            save_debug_image(
                                deglared_photo,
                                debug_dir / f"07_photo_{photo_idx:02d}_deglared.jpg",
                                f"Photo {photo_idx} after glare removal"
                            )

                        logger.debug(f"Photo {photo_idx}: removed glare (type={photo_glare.glare_type})")
                    else:
                        deglared_photo = photo
                        logger.debug(f"Photo {photo_idx}: no glare detected")

                    deglared_photos.append(deglared_photo)

                # Update extracted_photos with deglared versions
                extracted_photos = deglared_photos

                # Store overall stats (use first photo's glare detection for reporting)
                if len(extracted_photos) > 0:
                    glare_detection_result = detect_glare(
                        extracted_photos[0],
                        intensity_threshold=self.config.glare_intensity_threshold,
                        saturation_threshold=self.config.glare_saturation_threshold,
                        min_area=self.config.glare_min_area,
                        glare_type=self.config.glare_type,
                    )
                    glare_confidence_score = compute_glare_confidence(
                        extracted_photos[0], glare_detection_result.mask
                    )

                self.step_times['glare_detect'] = time.time() - step_start
                steps_completed.append('glare_detect')
                logger.info(
                    f"Glare removal time: {self.step_times['glare_detect']:.3f}s "
                    f"(processed {len(extracted_photos)} photos)"
                )

            except Exception as e:
                logger.warning(f"Glare removal failed, passing through: {e}")
                self.step_times['glare_detect'] = time.time() - step_start
                deglared_photos = extracted_photos  # Use originals

        # Output images are the deglared photos
        output_images = extracted_photos

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.3f}s")

        return PipelineResult(
            output_images=output_images,
            metadata=metadata,
            processing_time=total_time,
            steps_completed=steps_completed,
            page_detection=page_detection_result,
            glare_detection=glare_detection_result,
            glare_confidence=glare_confidence_score,
            glare_removal=None,  # Not tracked per-photo anymore
            photo_detections=photo_detections_result,
            num_photos_extracted=len(extracted_photos),
        )

    def process_batch(
        self,
        input_paths: List[str],
        debug_output_dir: Optional[str] = None,
        steps_filter: Optional[List[str]] = None
    ) -> List[PipelineResult]:
        """Process multiple images through the pipeline.

        Args:
            input_paths: List of paths to input images
            debug_output_dir: Optional directory for debug output
            steps_filter: Optional list of step names to run

        Returns:
            List of PipelineResult objects
        """
        results = []

        for i, path in enumerate(input_paths, 1):
            logger.info(f"Processing {i}/{len(input_paths)}: {path}")

            # Create separate debug directory for each image if debug is enabled
            if debug_output_dir:
                image_name = Path(path).stem
                debug_dir = Path(debug_output_dir) / image_name
            else:
                debug_dir = None

            try:
                result = self.process(
                    path,
                    debug_output_dir=str(debug_dir) if debug_dir else None,
                    steps_filter=steps_filter
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}", exc_info=True)
                continue

        return results


def _visualize_confidence(confidence_map: np.ndarray) -> np.ndarray:
    """Visualize confidence map as color gradient (green=high, red=low).

    Args:
        confidence_map: Confidence values [0, 1], shape (H, W)

    Returns:
        RGB image, float32 [0, 1], shape (H, W, 3)
    """
    import cv2

    # Convert confidence to hue: 0.0 (low conf) = red, 1.0 (high conf) = green
    # In HSV: red is 0°, green is 120°
    # Normalize to OpenCV HSV range: H in [0, 180]
    hue = (confidence_map * 60).astype(np.uint8)  # 0 to 60 (red to green in OpenCV HSV)

    # Full saturation and value
    saturation = np.full_like(hue, 255, dtype=np.uint8)
    value = np.full_like(hue, 255, dtype=np.uint8)

    # Create HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb.astype(np.float32) / 255.0
