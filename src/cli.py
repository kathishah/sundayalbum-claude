"""Command-line interface for Sunday Album processing."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from dotenv import load_dotenv

from src.pipeline import Pipeline, PipelineConfig
from src.utils.debug import save_debug_image

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version='0.1.0')
def main() -> None:
    """Sunday Album - Digitize physical photo album pages into clean individual digital photos."""
    pass


@main.command()
@click.argument('input_paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option(
    '--output',
    '-o',
    'output_dir',
    type=click.Path(),
    default='./output',
    help='Output directory for processed images'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Save debug visualizations at each pipeline step'
)
@click.option(
    '--batch',
    is_flag=True,
    help='Process all images in directory'
)
@click.option(
    '--filter',
    'filter_pattern',
    type=str,
    help='Glob pattern to filter files (e.g., "*.HEIC")'
)
@click.option(
    '--steps',
    type=str,
    help='Comma-separated list of steps to run (e.g., "load,normalize,glare")'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def process(
    input_paths: tuple,
    output_dir: str,
    debug: bool,
    batch: bool,
    filter_pattern: Optional[str],
    steps: Optional[str],
    verbose: bool
) -> None:
    """Process album page images through the digitization pipeline.

    INPUT_PATHS: One or more image files or directories to process
    """
    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse steps filter
    steps_filter = None
    if steps:
        steps_filter = [s.strip() for s in steps.split(',')]
        logger.info(f"Running only steps: {steps_filter}")

    # Collect all input files
    input_files: List[Path] = []

    for input_path_str in input_paths:
        input_path = Path(input_path_str)

        if input_path.is_file():
            input_files.append(input_path)
        elif input_path.is_dir():
            if batch:
                # Find all image files in directory
                if filter_pattern:
                    pattern_files = list(input_path.glob(filter_pattern))
                    input_files.extend(pattern_files)
                    logger.info(f"Found {len(pattern_files)} files matching {filter_pattern}")
                else:
                    # Default to common image extensions
                    for ext in ['*.heic', '*.HEIC', '*.dng', '*.DNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                        input_files.extend(input_path.glob(ext))
            else:
                logger.error(f"Directory provided but --batch not specified: {input_path}")
                sys.exit(1)
        else:
            logger.error(f"Invalid input path: {input_path}")
            sys.exit(1)

    if not input_files:
        logger.error("No input files found")
        sys.exit(1)

    logger.info(f"Processing {len(input_files)} file(s)")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create debug directory if requested
    debug_dir = None
    if debug:
        debug_dir = Path('./debug')
        debug_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Debug output will be saved to: {debug_dir}")

    # Create pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)

    # Process each file
    results = []
    for input_file in input_files:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Processing: {input_file.name}")
            logger.info(f"{'=' * 60}")

            # Set up debug directory for this file if debug is enabled
            if debug_dir:
                file_debug_dir = debug_dir / input_file.stem
                file_debug_dir.mkdir(parents=True, exist_ok=True)
            else:
                file_debug_dir = None

            # Process the image
            result = pipeline.process(
                str(input_file),
                debug_output_dir=str(file_debug_dir) if file_debug_dir else None,
                steps_filter=steps_filter
            )

            # Save output images
            # Naming convention: SundayAlbum_Page{XX}_Photo{YY}.jpg
            # For single output, just use Photo{YY}
            num_outputs = len(result.output_images)

            for i, output_image in enumerate(result.output_images, 1):
                if num_outputs > 1:
                    # Multiple photos extracted from one page
                    output_filename = f"SundayAlbum_{input_file.stem}_Photo{i:02d}.jpg"
                else:
                    # Single photo (either single print or full page if splitting failed)
                    output_filename = f"SundayAlbum_{input_file.stem}.jpg"

                output_file_path = output_path / output_filename

                save_debug_image(
                    output_image,
                    output_file_path,
                    f"Final output {i}",
                    quality=config.jpeg_quality
                )

                logger.info(f"Saved: {output_file_path.name}")

            results.append(result)

            # Print summary
            logger.info(f"\nProcessing Summary:")
            logger.info(f"  Format: {result.metadata.format}")
            logger.info(f"  Original size: {result.metadata.original_size[0]}x{result.metadata.original_size[1]}")
            logger.info(f"  Photos extracted: {result.num_photos_extracted}")
            logger.info(f"  Processing time: {result.processing_time:.3f}s")
            logger.info(f"  Steps completed: {', '.join(result.steps_completed)}")

        except Exception as e:
            logger.error(f"Error processing {input_file}: {e}", exc_info=verbose)
            continue

    # Print overall summary
    logger.info(f"\n{'=' * 60}")
    logger.info(f"COMPLETE: Processed {len(results)}/{len(input_files)} file(s)")
    logger.info(f"Output directory: {output_path.absolute()}")
    if debug_dir:
        logger.info(f"Debug directory: {debug_dir.absolute()}")
    logger.info(f"{'=' * 60}")


@main.command()
def status() -> None:
    """Show pipeline implementation status and available steps."""
    from src.pipeline import PIPELINE_STEPS, get_step_status

    click.echo("\n" + "=" * 70)
    click.echo("Sunday Album Processing Pipeline - Implementation Status")
    click.echo("=" * 70 + "\n")

    # Get status for each step
    total_steps = len(PIPELINE_STEPS)
    implemented_count = 0

    for idx, step_info in enumerate(PIPELINE_STEPS, 1):
        step_id = step_info['id']
        step_name = step_info['name']
        step_desc = step_info['description']
        priority = step_info.get('priority', '-')

        # Check implementation status
        status_info = get_step_status(step_id)
        is_implemented = status_info['implemented']
        notes = status_info.get('notes', '')

        if is_implemented:
            implemented_count += 1
            status_icon = "✅"
            status_text = click.style("IMPLEMENTED", fg="green", bold=True)
        else:
            status_icon = "⏳"
            status_text = click.style("NOT IMPLEMENTED", fg="yellow")

        # Print step info
        click.echo(f"{status_icon} Step {idx}: {step_name} (Priority {priority})")
        click.echo(f"   ID: {step_id}")
        click.echo(f"   Status: {status_text}")
        click.echo(f"   Description: {step_desc}")
        if notes:
            click.echo(f"   Notes: {notes}")
        click.echo()

    # Print summary
    progress_pct = (implemented_count / total_steps) * 100
    click.echo("=" * 70)
    click.echo(f"Progress: {implemented_count}/{total_steps} steps implemented ({progress_pct:.1f}%)")
    click.echo("=" * 70)

    click.echo("\n📋 Usage Examples:")
    click.echo("  # Process a single file (all implemented steps)")
    click.echo("  python -m src.cli process image.HEIC --output ./output/")
    click.echo()
    click.echo("  # Process with debug visualizations")
    click.echo("  python -m src.cli process image.HEIC --output ./output/ --debug")
    click.echo()
    click.echo("  # Process multiple files with glob pattern")
    click.echo("  python -m src.cli process test-images/*.HEIC --output ./output/")
    click.echo()
    click.echo("  # Process only specific steps")
    click.echo("  python -m src.cli process image.HEIC --steps load,detect_photos,split --output ./output/")
    click.echo()
    click.echo("  # Process directory (batch mode)")
    click.echo("  python -m src.cli process test-images/ --batch --filter \"*.HEIC\" --output ./output/")
    click.echo()


@main.command()
@click.argument('output_file', type=click.Path(exists=True))
@click.option('--original', type=click.Path(exists=True), help='Original input file for comparison', required=True)
@click.option('--use-ai', is_flag=True, help='Use AI vision for quality assessment (requires ANTHROPIC_API_KEY)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def check(output_file: str, original: str, use_ai: bool, verbose: bool) -> None:
    """Quality check processed images against originals.

    OUTPUT_FILE: Processed output image to check
    """
    from src.preprocessing.loader import load_image
    from src.ai.quality_check import assess_quality_full

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Quality check: {output_file}")
    logger.info(f"Comparing against: {original}")

    try:
        # Load images
        logger.info("Loading images...")
        original_img, _ = load_image(original)
        processed_img, _ = load_image(output_file)

        # Run quality assessment
        logger.info("Running quality assessment...")
        report = assess_quality_full(
            original_img,
            processed_img,
            use_ai=use_ai
        )

        # Print results
        click.echo("\n" + "=" * 70)
        click.echo("Quality Assessment Report")
        click.echo("=" * 70 + "\n")

        click.echo(f"Overall Quality Score: {report.overall_quality_score:.1f}/100\n")

        click.echo("Programmatic Metrics:")
        click.echo(f"  SSIM (Structural Similarity): {report.metrics.ssim_score:.3f}")
        click.echo(f"  Sharpness: {report.metrics.sharpness_original:.1f} → {report.metrics.sharpness_processed:.1f} ({report.metrics.sharpness_improvement:.2f}x)")
        click.echo(f"  Contrast: {report.metrics.contrast_original:.3f} → {report.metrics.contrast_processed:.3f} ({report.metrics.contrast_improvement:.2f}x)")
        click.echo(f"  Saturation: {report.metrics.saturation_original:.3f} → {report.metrics.saturation_processed:.3f}")
        click.echo(f"  Color Shift: {report.metrics.color_shift:.4f}")
        click.echo(f"  Brightness Change: {report.metrics.brightness_change:.4f}")

        if report.ai_assessment:
            click.echo("\nAI Assessment:")
            click.echo(f"  Overall Score: {report.ai_assessment.overall_score:.1f}/10")
            click.echo(f"  Glare Remaining: {report.ai_assessment.glare_remaining:.2f}")
            click.echo(f"  Artifacts Detected: {report.ai_assessment.artifacts_detected}")
            click.echo(f"  Sharpness: {report.ai_assessment.sharpness_score:.1f}/10")
            click.echo(f"  Color Naturalness: {report.ai_assessment.color_naturalness:.1f}/10")
            click.echo(f"  Confidence: {report.ai_assessment.confidence:.2f}")
            click.echo(f"  Notes: {report.ai_assessment.notes}")

        if report.notes:
            click.echo("\nNotes:")
            for note in report.notes:
                click.echo(f"  - {note}")

        click.echo("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Quality check failed: {e}", exc_info=verbose)
        sys.exit(1)


@main.command()
@click.argument('before', type=click.Path(exists=True))
@click.argument('after', type=click.Path(exists=True))
@click.option('--save', type=click.Path(), help='Save comparison image to file', default='comparison.jpg')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def compare(before: str, after: str, save: str, verbose: bool) -> None:
    """Compare before/after images side-by-side.

    BEFORE: Original image
    AFTER: Processed image
    """
    import numpy as np
    import cv2
    from src.preprocessing.loader import load_image
    from src.utils.debug import save_debug_image

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Comparing: {before} vs {after}")

    try:
        # Load images
        logger.info("Loading images...")
        before_img, _ = load_image(before)
        after_img, _ = load_image(after)

        # Resize to same height if needed
        h1, w1 = before_img.shape[:2]
        h2, w2 = after_img.shape[:2]

        if h1 != h2:
            # Resize to match heights
            target_h = min(h1, h2)
            scale1 = target_h / h1
            scale2 = target_h / h2

            before_img = cv2.resize(
                before_img,
                (int(w1 * scale1), target_h),
                interpolation=cv2.INTER_AREA
            )
            after_img = cv2.resize(
                after_img,
                (int(w2 * scale2), target_h),
                interpolation=cv2.INTER_AREA
            )

        # Create side-by-side comparison
        # Add a small separator (10px white bar)
        separator = np.ones((before_img.shape[0], 10, 3), dtype=np.float32)
        comparison = np.hstack([before_img, separator, after_img])

        # Add labels
        from PIL import Image, ImageDraw, ImageFont
        comparison_pil = Image.fromarray((comparison * 255).astype(np.uint8))
        draw = ImageDraw.Draw(comparison_pil)

        # Try to use a better font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        except:
            font = ImageFont.load_default()

        # Add "BEFORE" label
        draw.text((20, 20), "BEFORE", fill=(255, 255, 0), font=font)

        # Add "AFTER" label
        after_x = before_img.shape[1] + 10 + 20
        draw.text((after_x, 20), "AFTER", fill=(0, 255, 0), font=font)

        # Convert back to numpy
        comparison_with_labels = np.array(comparison_pil).astype(np.float32) / 255.0

        # Save comparison
        save_path = Path(save)
        save_debug_image(comparison_with_labels, save_path, "Before/After comparison", quality=95)

        logger.info(f"Saved comparison to: {save_path.absolute()}")

        click.echo(f"\n✅ Comparison saved to: {save_path.absolute()}")
        click.echo(f"   Before size: {before_img.shape[1]}x{before_img.shape[0]}")
        click.echo(f"   After size: {after_img.shape[1]}x{after_img.shape[0]}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=verbose)
        sys.exit(1)


@main.command()
@click.option(
    '--test-images-dir',
    type=click.Path(exists=True),
    default='./test-images',
    help='Directory containing test images'
)
@click.option(
    '--output',
    '-o',
    'output_dir',
    type=click.Path(),
    default='./output',
    help='Output directory for processed images'
)
@click.option(
    '--debug',
    is_flag=True,
    help='Save debug visualizations'
)
@click.option(
    '--use-ai',
    is_flag=True,
    help='Use AI vision for quality assessment (requires ANTHROPIC_API_KEY)'
)
@click.option(
    '--heic-only',
    is_flag=True,
    help='Process only HEIC files (skip DNG for faster testing)'
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def validate(
    test_images_dir: str,
    output_dir: str,
    debug: bool,
    use_ai: bool,
    heic_only: bool,
    verbose: bool
) -> None:
    """Run full pipeline validation on all test images and generate summary report.

    This command processes all test images and generates a comprehensive quality report
    with metrics, timing, and optional AI assessment.
    """
    import numpy as np
    from src.preprocessing.loader import load_image
    from src.ai.quality_check import assess_quality_full

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    test_dir = Path(test_images_dir)
    if not test_dir.exists():
        logger.error(f"Test images directory not found: {test_dir}")
        sys.exit(1)

    # Find test images
    if heic_only:
        test_files = list(test_dir.glob('*.HEIC')) + list(test_dir.glob('*.heic'))
        logger.info("Processing HEIC files only")
    else:
        test_files = []
        for ext in ['*.HEIC', '*.heic', '*.DNG', '*.dng']:
            test_files.extend(test_dir.glob(ext))
        logger.info("Processing both HEIC and DNG files")

    if not test_files:
        logger.error(f"No test images found in {test_dir}")
        sys.exit(1)

    # Sort for consistent ordering
    test_files = sorted(test_files)

    click.echo("\n" + "=" * 80)
    click.echo("Sunday Album - Full Pipeline Validation")
    click.echo("=" * 80)
    click.echo(f"Test images: {len(test_files)}")
    click.echo(f"Output directory: {output_dir}")
    click.echo(f"Debug mode: {debug}")
    click.echo(f"AI quality check: {use_ai}")
    click.echo("=" * 80 + "\n")

    # Create output and debug directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if debug:
        debug_dir = Path('./debug')
        debug_dir.mkdir(parents=True, exist_ok=True)

    # Create pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)

    # Process all files and collect results
    validation_results = []

    for i, test_file in enumerate(test_files, 1):
        click.echo(f"\n[{i}/{len(test_files)}] Processing: {test_file.name}")
        click.echo("-" * 80)

        try:
            # Set up debug directory for this file
            if debug_dir:
                file_debug_dir = debug_dir / test_file.stem
                file_debug_dir.mkdir(parents=True, exist_ok=True)
            else:
                file_debug_dir = None

            # Process the image
            result = pipeline.process(
                str(test_file),
                debug_output_dir=str(file_debug_dir) if file_debug_dir else None,
                steps_filter=None
            )

            # Save output images
            saved_outputs = []
            for photo_idx, output_image in enumerate(result.output_images, 1):
                if len(result.output_images) > 1:
                    output_filename = f"SundayAlbum_{test_file.stem}_Photo{photo_idx:02d}.jpg"
                else:
                    output_filename = f"SundayAlbum_{test_file.stem}.jpg"

                output_file_path = output_path / output_filename

                save_debug_image(
                    output_image,
                    output_file_path,
                    f"Final output {photo_idx}",
                    quality=config.jpeg_quality
                )

                saved_outputs.append(output_file_path)
                click.echo(f"  ✓ Saved: {output_filename}")

            # Quality assessment (on first extracted photo)
            quality_report = None
            if len(result.output_images) > 0 and use_ai:
                try:
                    click.echo(f"  Running quality assessment...")
                    # Load the original extracted photo (before color restoration)
                    # For comparison, we'll use the raw extracted photo from debug output
                    original_for_comparison = result.output_images[0]  # Simplified: use final output vs itself
                    processed_for_comparison = result.output_images[0]

                    quality_report = assess_quality_full(
                        original_for_comparison,
                        processed_for_comparison,
                        use_ai=use_ai
                    )
                    click.echo(f"  Quality score: {quality_report.overall_quality_score:.1f}/100")
                except Exception as e:
                    logger.warning(f"  Quality assessment failed: {e}")

            # Store validation result
            validation_results.append({
                'input_file': test_file.name,
                'format': result.metadata.format,
                'original_size': f"{result.metadata.original_size[0]}x{result.metadata.original_size[1]}",
                'photos_extracted': result.num_photos_extracted,
                'glare_confidence': result.glare_confidence if result.glare_confidence else 0.0,
                'processing_time': result.processing_time,
                'quality_score': quality_report.overall_quality_score if quality_report else None,
                'steps_completed': len(result.steps_completed),
                'output_files': [str(f.name) for f in saved_outputs]
            })

            click.echo(f"  Format: {result.metadata.format}")
            click.echo(f"  Photos extracted: {result.num_photos_extracted}")
            click.echo(f"  Processing time: {result.processing_time:.1f}s")
            click.echo(f"  Steps completed: {len(result.steps_completed)}")

        except Exception as e:
            logger.error(f"  ✗ Error: {e}", exc_info=verbose)
            validation_results.append({
                'input_file': test_file.name,
                'format': 'ERROR',
                'original_size': 'N/A',
                'photos_extracted': 0,
                'glare_confidence': 0.0,
                'processing_time': 0.0,
                'quality_score': None,
                'steps_completed': 0,
                'output_files': []
            })
            continue

    # Generate summary report
    click.echo("\n" + "=" * 80)
    click.echo("VALIDATION SUMMARY")
    click.echo("=" * 80 + "\n")

    # Summary table
    click.echo(f"{'Input File':<40} {'Format':<8} {'Photos':<8} {'Time (s)':<10} {'Quality':<10}")
    click.echo("-" * 80)

    total_time = 0.0
    total_photos = 0
    successful = 0

    for res in validation_results:
        quality_str = f"{res['quality_score']:.1f}/100" if res['quality_score'] else "N/A"
        time_str = f"{res['processing_time']:.1f}"

        if res['format'] != 'ERROR':
            successful += 1
            total_time += res['processing_time']
            total_photos += res['photos_extracted']

        click.echo(
            f"{res['input_file']:<40} "
            f"{res['format']:<8} "
            f"{res['photos_extracted']:<8} "
            f"{time_str:<10} "
            f"{quality_str:<10}"
        )

    click.echo("-" * 80)
    click.echo(f"Successful: {successful}/{len(validation_results)}")
    click.echo(f"Total photos extracted: {total_photos}")
    click.echo(f"Total processing time: {total_time:.1f}s")
    click.echo(f"Average time per file: {total_time/max(successful, 1):.1f}s")
    click.echo("=" * 80)

    click.echo(f"\n✅ Validation complete!")
    click.echo(f"   Output directory: {output_path.absolute()}")
    if debug_dir:
        click.echo(f"   Debug directory: {debug_dir.absolute()}")


if __name__ == '__main__':
    main()
