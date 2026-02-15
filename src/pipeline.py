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
    glare_type: str = "auto"  # "auto", "sleeve" (flat plastic), or "print" (curved glossy)

    # Photo detection
    photo_detect_method: str = "contour"  # "contour", "yolo", or "claude"
    photo_detect_min_area_ratio: float = 0.05
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

        # Step 4: Glare detection
        glare_detection_result: Optional[GlareDetection] = None
        glare_confidence_score: Optional[float] = None
        should_run_glare_detect = steps_filter is None or 'glare' in steps_filter

        if should_run_glare_detect:
            step_start = time.time()
            try:
                glare_detection_result = detect_glare(
                    working_image,
                    intensity_threshold=self.config.glare_intensity_threshold,
                    saturation_threshold=self.config.glare_saturation_threshold,
                    min_area=self.config.glare_min_area,
                    glare_type=self.config.glare_type,
                )

                # Compute confidence
                glare_confidence_score = compute_glare_confidence(
                    working_image, glare_detection_result.mask
                )

                # Save debug outputs
                if debug_dir:
                    from src.utils.debug import save_debug_image, save_debug_text
                    import cv2

                    # 04_glare_mask.png — binary mask (white = detected glare)
                    save_debug_image(
                        glare_detection_result.mask,
                        debug_dir / "04_glare_mask.png",
                        "Detected glare regions (binary mask)"
                    )

                    # 05_glare_overlay.jpg — original with semi-transparent overlay
                    overlay = draw_glare_overlay(working_image, glare_detection_result)
                    save_debug_image(
                        overlay,
                        debug_dir / "05_glare_overlay.jpg",
                        f"Glare overlay (type={glare_detection_result.glare_type}, "
                        f"area={glare_detection_result.total_glare_area_ratio:.3f})"
                    )

                    # 05_glare_type.txt — classified glare type
                    save_debug_text(
                        f"Detected glare type: {glare_detection_result.glare_type}\n"
                        f"Total glare area ratio: {glare_detection_result.total_glare_area_ratio:.4f}\n"
                        f"Number of regions: {len(glare_detection_result.regions)}\n"
                        f"Confidence score: {glare_confidence_score:.4f}\n",
                        debug_dir / "05_glare_type.txt"
                    )

                self.step_times['glare_detect'] = time.time() - step_start
                steps_completed.append('glare_detect')
                logger.info(
                    f"Glare detection time: {self.step_times['glare_detect']:.3f}s, "
                    f"type={glare_detection_result.glare_type}, "
                    f"confidence={glare_confidence_score:.3f}"
                )

            except Exception as e:
                logger.warning(f"Glare detection failed, passing through: {e}")
                self.step_times['glare_detect'] = time.time() - step_start

        # Current output is the working image (after page detection + perspective correction)
        output_images = [working_image]

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
