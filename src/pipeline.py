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
from src.geometry import correct_keystone, correct_rotation, correct_warp
from src.color import (
    auto_white_balance,
    remove_yellowing_adaptive,
    restore_fading,
    enhance_adaptive,
)

logger = logging.getLogger(__name__)


# Pipeline step definitions with implementation status
PIPELINE_STEPS = [
    {
        'id': 'load',
        'name': 'Load Image',
        'description': 'Load HEIC, DNG, JPEG, or PNG image with format detection',
        'priority': 1,
        'implemented': True,
    },
    {
        'id': 'normalize',
        'name': 'Normalize & Preprocess',
        'description': 'Resize, apply EXIF orientation, generate working copy',
        'priority': 1,
        'implemented': True,
    },
    {
        'id': 'page_detect',
        'name': 'Page Detection',
        'description': 'Detect album page boundaries and apply perspective correction',
        'priority': 3,
        'implemented': True,
    },
    {
        'id': 'glare_detect',
        'name': 'Glare Detection',
        'description': 'Detect specular highlights from plastic sleeves or glossy prints',
        'priority': 1,
        'implemented': True,
    },
    {
        'id': 'glare_remove',
        'name': 'Glare Removal',
        'description': 'Inpaint or composite glare regions',
        'priority': 1,
        'implemented': False,
        'notes': 'Detection complete, removal partially implemented (single-shot only)',
    },
    {
        'id': 'photo_detect',
        'name': 'Photo Detection',
        'description': 'Detect individual photo boundaries on album pages',
        'priority': 2,
        'implemented': True,
    },
    {
        'id': 'photo_split',
        'name': 'Photo Splitting',
        'description': 'Extract individual photos as separate images',
        'priority': 2,
        'implemented': True,
    },
    {
        'id': 'keystone_correct',
        'name': 'Keystone Correction',
        'description': 'Per-photo perspective correction for tilted shots',
        'priority': 3,
        'implemented': True,
    },
    {
        'id': 'dewarp',
        'name': 'Dewarping',
        'description': 'Correct bulge/curvature from bowed photo surfaces',
        'priority': 3,
        'implemented': True,
    },
    {
        'id': 'rotation_correct',
        'name': 'Rotation Correction',
        'description': 'Auto-detect and correct rotation angle',
        'priority': 3,
        'implemented': True,
    },
    {
        'id': 'white_balance',
        'name': 'White Balance',
        'description': 'Auto white balance correction',
        'priority': 4,
        'implemented': True,
    },
    {
        'id': 'color_restore',
        'name': 'Color Restoration',
        'description': 'Fade restoration with CLAHE, saturation boost',
        'priority': 4,
        'implemented': True,
    },
    {
        'id': 'deyellow',
        'name': 'Deyellowing',
        'description': 'Remove yellowing from aged photos',
        'priority': 4,
        'implemented': True,
    },
    {
        'id': 'sharpen',
        'name': 'Sharpening',
        'description': 'Unsharp mask and final enhancement',
        'priority': 4,
        'implemented': True,
    },
]


def get_step_status(step_id: str) -> dict:
    """Get implementation status for a pipeline step.

    Args:
        step_id: Step identifier (e.g., 'load', 'normalize', 'glare_detect')

    Returns:
        Dictionary with 'implemented' (bool) and optional 'notes' (str)
    """
    for step in PIPELINE_STEPS:
        if step['id'] == step_id:
            return {
                'implemented': step['implemented'],
                'notes': step.get('notes', '')
            }

    return {'implemented': False, 'notes': 'Unknown step'}


@dataclass
class PipelineConfig:
    """All tunable parameters in one place."""

    # Preprocessing
    max_working_resolution: int = 4000  # px, longest edge
    prefer_heic_for_iteration: bool = True

    # Page detection (GrabCut)
    page_detect_min_area_ratio: float = 0.3
    page_detect_grabcut_iterations: int = 5
    page_detect_grabcut_max_dimension: int = 800

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
    dewarp_detection_threshold: float = 0.02  # Minimum curvature ratio (2%)

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

    # AI orientation correction (Step 4.5)
    use_ai_orientation: bool = True
    ai_orientation_model: str = "claude-haiku-4-5-20251001"
    ai_orientation_min_confidence: str = "medium"  # ignore "low" confidence detections

    # OpenAI glare removal (opt-in; OpenCV is still the default)
    use_openai_glare_removal: bool = False
    openai_model: str = "gpt-image-1.5"
    openai_glare_quality: str = "high"
    openai_glare_input_fidelity: str = "high"
    # If set, overrides the Claude-generated scene description for the glare prompt.
    # Orientation analysis still runs; only the description is replaced.
    forced_scene_description: Optional[str] = None


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
                    min_area_ratio=self.config.page_detect_min_area_ratio,
                    grabcut_iterations=self.config.page_detect_grabcut_iterations,
                    grabcut_max_dimension=self.config.page_detect_grabcut_max_dimension,
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
                # Detect individual photos on the page.
                # page_was_corrected=True tells the detector that the image has
                # already been perspective-corrected by GrabCut page detection,
                # so it can apply the album-border pre-check to avoid false splits
                # on single prints (cave/harbor/skydiving).
                _page_was_corrected = (
                    page_detection_result is not None
                    and not page_detection_result.is_full_frame
                )
                photo_detections_result = detect_photos(
                    working_image,
                    min_area_ratio=self.config.photo_detect_min_area_ratio,
                    max_count=self.config.photo_detect_max_count,
                    method=self.config.photo_detect_method,
                    page_was_corrected=_page_was_corrected,
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

        # Step 4.5: AI orientation correction (per photo)
        # Detects gross 90°/180°/270° errors that heuristic rotation cannot handle.
        # Also produces a scene description used by the OpenAI glare step.
        photo_analyses: list = []  # parallel to extracted_photos
        should_run_ai_orientation = (
            self.config.use_ai_orientation
            and (steps_filter is None or 'ai_orientation' in steps_filter)
        )

        if should_run_ai_orientation:
            step_start = time.time()
            from src.ai.claude_vision import analyze_photo_for_processing, apply_orientation
            from src.utils.secrets import load_secrets

            secrets = load_secrets()
            anthropic_key = secrets.anthropic_api_key

            if not anthropic_key:
                logger.warning("ANTHROPIC_API_KEY not set; skipping AI orientation step")
                photo_analyses = [None] * len(extracted_photos)
            else:
                oriented_photos: List[np.ndarray] = []
                for photo_idx, photo in enumerate(extracted_photos, 1):
                    logger.debug(
                        f"AI orientation: analysing photo {photo_idx}/{len(extracted_photos)}"
                    )
                    analysis = analyze_photo_for_processing(
                        photo,
                        api_key=anthropic_key,
                        model=self.config.ai_orientation_model,
                    )
                    photo_analyses.append(analysis)

                    # Apply correction only if confidence meets threshold
                    if analysis.orientation_confidence in ("medium", "high"):
                        corrected = apply_orientation(photo, analysis)
                        logger.info(
                            f"Photo {photo_idx}: orientation corrected "
                            f"({analysis.rotation_degrees}°, "
                            f"flip={analysis.flip_horizontal}, "
                            f"confidence={analysis.orientation_confidence})"
                        )
                        if debug_dir:
                            from src.utils.debug import save_debug_image
                            save_debug_image(
                                corrected,
                                debug_dir / f"05b_photo_{photo_idx:02d}_oriented.jpg",
                                f"Photo {photo_idx} after orientation correction "
                                f"({analysis.rotation_degrees}°)"
                            )
                    else:
                        corrected = photo
                        logger.debug(
                            f"Photo {photo_idx}: orientation confidence=low, skipping"
                        )
                    oriented_photos.append(corrected)

                extracted_photos = oriented_photos
                self.step_times['ai_orientation'] = time.time() - step_start
                steps_completed.append('ai_orientation')
                logger.info(
                    f"AI orientation time: {self.step_times['ai_orientation']:.3f}s "
                    f"({len(extracted_photos)} photos)"
                )
        else:
            photo_analyses = [None] * len(extracted_photos)

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
                        if self.config.use_openai_glare_removal:
                            # Use OpenAI diffusion-based inpainting (opt-in)
                            from src.glare.remover_openai import remove_glare_openai
                            from src.utils.secrets import load_secrets
                            secrets = load_secrets()
                            openai_key = secrets.openai_api_key

                            if openai_key:
                                # Get scene description: forced override > orientation analysis > default
                                analysis = photo_analyses[photo_idx - 1] if photo_analyses else None
                                scene_desc = (
                                    self.config.forced_scene_description
                                    or (analysis.scene_description if analysis and analysis.scene_description else "")
                                    or "A printed photograph."
                                )
                                deglared_photo = remove_glare_openai(
                                    photo,
                                    scene_desc=scene_desc,
                                    api_key=openai_key,
                                    model=self.config.openai_model,
                                    quality=self.config.openai_glare_quality,
                                    input_fidelity=self.config.openai_glare_input_fidelity,
                                )
                                logger.debug(
                                    f"Photo {photo_idx}: OpenAI glare removal applied"
                                )
                            else:
                                logger.warning(
                                    "OPENAI_API_KEY not set; falling back to OpenCV inpainting"
                                )
                                glare_result = remove_glare_single(
                                    photo,
                                    photo_glare.mask,
                                    photo_glare.severity_map,
                                    inpaint_radius=self.config.glare_inpaint_radius,
                                    feather_radius=self.config.glare_feather_radius,
                                )
                                deglared_photo = glare_result.image
                        else:
                            # Default: OpenCV inpainting
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

        # Step 6: Geometry correction (per photo)
        # Apply keystone correction, rotation correction, and dewarping
        should_run_geometry = (
            steps_filter is None or
            'keystone_correct' in steps_filter or
            'rotation_correct' in steps_filter or
            'dewarp' in steps_filter
        )

        if should_run_geometry:
            step_start = time.time()
            geometry_corrected_photos: List[np.ndarray] = []

            try:
                for photo_idx, photo in enumerate(extracted_photos, 1):
                    logger.debug(f"Processing geometry for photo {photo_idx}/{len(extracted_photos)}")

                    corrected_photo = photo
                    corrections_applied = []

                    # 1. Rotation correction (do this first)
                    if steps_filter is None or 'rotation_correct' in steps_filter:
                        corrected_photo, rotation_angle = correct_rotation(
                            corrected_photo,
                            auto_correct_max=self.config.rotation_auto_correct_max,
                        )
                        if rotation_angle != 0:
                            corrections_applied.append(f"rotation: {rotation_angle:.2f}°")
                            logger.debug(f"Photo {photo_idx}: applied rotation {rotation_angle:.2f}°")

                            if debug_dir:
                                from src.utils.debug import save_debug_image
                                save_debug_image(
                                    corrected_photo,
                                    debug_dir / f"08_photo_{photo_idx:02d}_rotated_{rotation_angle:.1f}deg.jpg",
                                    f"Photo {photo_idx} after rotation correction ({rotation_angle:.2f}°)"
                                )

                    # 2. Keystone correction (after rotation)
                    # Note: We don't have corners here since photos are already extracted
                    # Keystone correction would primarily be useful if we had corner info
                    # Skip for now since splitter.py already does perspective correction
                    # This is here for potential future use with re-detection
                    if steps_filter is None or 'keystone_correct' in steps_filter:
                        # Keystone correction is already applied during photo extraction
                        # in splitter.py, so we skip it here to avoid double-correction
                        pass

                    # 3. Dewarp correction (correct barrel distortion)
                    if steps_filter is None or 'dewarp' in steps_filter:
                        corrected_photo, warp_detected = correct_warp(corrected_photo)
                        if warp_detected:
                            corrections_applied.append("dewarp")
                            logger.debug(f"Photo {photo_idx}: applied dewarp correction")

                            if debug_dir:
                                from src.utils.debug import save_debug_image
                                save_debug_image(
                                    corrected_photo,
                                    debug_dir / f"09_photo_{photo_idx:02d}_dewarped.jpg",
                                    f"Photo {photo_idx} after dewarp correction"
                                )

                    # Save final geometry-corrected photo
                    if debug_dir and corrections_applied:
                        from src.utils.debug import save_debug_image
                        save_debug_image(
                            corrected_photo,
                            debug_dir / f"10_photo_{photo_idx:02d}_geometry_final.jpg",
                            f"Photo {photo_idx} after geometry corrections: {', '.join(corrections_applied)}"
                        )

                    geometry_corrected_photos.append(corrected_photo)

                # Update extracted_photos with geometry-corrected versions
                extracted_photos = geometry_corrected_photos

                self.step_times['geometry'] = time.time() - step_start
                steps_completed.append('geometry')
                logger.info(
                    f"Geometry correction time: {self.step_times['geometry']:.3f}s "
                    f"(processed {len(extracted_photos)} photos)"
                )

            except Exception as e:
                logger.warning(f"Geometry correction failed, passing through: {e}")
                self.step_times['geometry'] = time.time() - step_start

        # Step 7: Color restoration (per photo)
        # Apply white balance, deyellowing, fade restoration, and sharpening
        should_run_color = (
            steps_filter is None or
            'white_balance' in steps_filter or
            'deyellow' in steps_filter or
            'color_restore' in steps_filter or
            'sharpen' in steps_filter
        )

        if should_run_color:
            step_start = time.time()
            color_restored_photos: List[np.ndarray] = []

            try:
                for photo_idx, photo in enumerate(extracted_photos, 1):
                    logger.debug(f"Processing color restoration for photo {photo_idx}/{len(extracted_photos)}")

                    restored_photo = photo
                    restorations_applied = []

                    # 1. White balance correction
                    if steps_filter is None or 'white_balance' in steps_filter:
                        restored_photo, wb_info = auto_white_balance(
                            restored_photo,
                            page_border=None,
                            method="gray_world"
                        )
                        restorations_applied.append(f"white_balance ({wb_info['method_used']})")
                        logger.debug(
                            f"Photo {photo_idx}: white balance applied "
                            f"(R={wb_info.get('gain_r', 1.0):.2f}, "
                            f"G={wb_info.get('gain_g', 1.0):.2f}, "
                            f"B={wb_info.get('gain_b', 1.0):.2f})"
                        )

                        if debug_dir:
                            from src.utils.debug import save_debug_image
                            save_debug_image(
                                restored_photo,
                                debug_dir / f"11_photo_{photo_idx:02d}_wb.jpg",
                                f"Photo {photo_idx} after white balance"
                            )

                    # 2. Deyellowing (adaptive)
                    if steps_filter is None or 'deyellow' in steps_filter:
                        restored_photo, deyellow_info = remove_yellowing_adaptive(restored_photo)
                        if deyellow_info['corrected']:
                            restorations_applied.append(
                                f"deyellow (score={deyellow_info['yellowing_score']:.2f}, "
                                f"shift={deyellow_info['shift_applied']:.1f})"
                            )
                            logger.debug(
                                f"Photo {photo_idx}: yellowing removed "
                                f"(score={deyellow_info['yellowing_score']:.3f})"
                            )

                            if debug_dir:
                                from src.utils.debug import save_debug_image
                                save_debug_image(
                                    restored_photo,
                                    debug_dir / f"12_photo_{photo_idx:02d}_deyellow.jpg",
                                    f"Photo {photo_idx} after deyellowing"
                                )

                    # 3. Fade restoration (CLAHE + saturation boost)
                    if steps_filter is None or 'color_restore' in steps_filter:
                        restored_photo, restore_info = restore_fading(
                            restored_photo,
                            clahe_clip_limit=self.config.clahe_clip_limit,
                            clahe_grid_size=self.config.clahe_grid_size,
                            saturation_boost=self.config.saturation_boost,
                            auto_detect_fading=True
                        )
                        restorations_applied.append(
                            f"fade_restore (contrast={restore_info['contrast_improvement']:.2f}x, "
                            f"sat_boost={restore_info['saturation_boost_applied']:.2f})"
                        )
                        logger.debug(
                            f"Photo {photo_idx}: fade restoration applied "
                            f"(contrast improvement={restore_info['contrast_improvement']:.2f}x)"
                        )

                        if debug_dir:
                            from src.utils.debug import save_debug_image
                            save_debug_image(
                                restored_photo,
                                debug_dir / f"13_photo_{photo_idx:02d}_restored.jpg",
                                f"Photo {photo_idx} after fade restoration"
                            )

                    # 4. Sharpening and final enhancement (adaptive)
                    if steps_filter is None or 'sharpen' in steps_filter:
                        restored_photo, enhance_info = enhance_adaptive(restored_photo)
                        restorations_applied.append(
                            f"sharpen (amount={enhance_info['sharpen_amount']:.2f}, "
                            f"sharpness={enhance_info['sharpness_improvement']:.2f}x)"
                        )
                        logger.debug(
                            f"Photo {photo_idx}: sharpening applied "
                            f"(sharpness improvement={enhance_info['sharpness_improvement']:.2f}x)"
                        )

                        if debug_dir:
                            from src.utils.debug import save_debug_image
                            save_debug_image(
                                restored_photo,
                                debug_dir / f"14_photo_{photo_idx:02d}_enhanced.jpg",
                                f"Photo {photo_idx} after enhancement"
                            )

                    # Save final color-restored photo
                    if debug_dir and restorations_applied:
                        from src.utils.debug import save_debug_image
                        save_debug_image(
                            restored_photo,
                            debug_dir / f"15_photo_{photo_idx:02d}_final.jpg",
                            f"Photo {photo_idx} final output: {', '.join(restorations_applied)}"
                        )

                    color_restored_photos.append(restored_photo)

                # Update extracted_photos with color-restored versions
                extracted_photos = color_restored_photos

                self.step_times['color_restoration'] = time.time() - step_start
                steps_completed.append('color_restoration')
                logger.info(
                    f"Color restoration time: {self.step_times['color_restoration']:.3f}s "
                    f"(processed {len(extracted_photos)} photos)"
                )

            except Exception as e:
                logger.warning(f"Color restoration failed, passing through: {e}")
                self.step_times['color_restoration'] = time.time() - step_start

        # Output images are the fully processed photos
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
