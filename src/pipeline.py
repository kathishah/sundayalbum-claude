"""Main pipeline orchestrator for Sunday Album processing."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.preprocessing.loader import load_image, ImageMetadata
from src.preprocessing.normalizer import normalize, NormalizationResult

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
        steps_completed = []

        logger.info(f"Processing: {input_path}")

        # Step 1: Load image
        step_start = time.time()
        image, metadata = load_image(input_path)
        self.step_times['load'] = time.time() - step_start
        steps_completed.append('load')
        logger.info(f"Load time: {self.step_times['load']:.3f}s")

        # Save debug output if requested
        if debug_output_dir:
            from src.utils.debug import save_debug_image
            debug_dir = Path(debug_output_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)
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

        # Save debug output if requested
        if debug_output_dir:
            from src.utils.debug import save_debug_image
            debug_dir = Path(debug_output_dir)
            save_debug_image(
                norm_result.image,
                debug_dir / "02_normalized.jpg",
                f"Normalized to {norm_result.image.shape[1]}x{norm_result.image.shape[0]}"
            )
            save_debug_image(
                norm_result.thumbnail,
                debug_dir / "02_thumbnail.jpg",
                f"Thumbnail {norm_result.thumbnail.shape[1]}x{norm_result.thumbnail.shape[0]}"
            )

        # For Phase 1, we just return the normalized image
        # Future phases will add more steps here
        output_images = [norm_result.image]

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.3f}s")

        return PipelineResult(
            output_images=output_images,
            metadata=metadata,
            processing_time=total_time,
            steps_completed=steps_completed
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
