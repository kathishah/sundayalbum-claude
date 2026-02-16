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
@click.option('--original', type=click.Path(exists=True), help='Original input file for comparison')
def check(output_file: str, original: Optional[str]) -> None:
    """Quality check using AI vision (placeholder for Phase 9).

    OUTPUT_FILE: Processed output image to check
    """
    logger.info(f"Quality check: {output_file}")
    if original:
        logger.info(f"Comparing against: {original}")
    logger.info("AI quality check not yet implemented (coming in Phase 9)")


@main.command()
@click.argument('before', type=click.Path(exists=True))
@click.argument('after', type=click.Path(exists=True))
@click.option('--save', type=click.Path(), help='Save comparison image to file')
def compare(before: str, after: str, save: Optional[str]) -> None:
    """Compare before/after images side-by-side (placeholder for Phase 8).

    BEFORE: Original image
    AFTER: Processed image
    """
    logger.info(f"Comparing: {before} vs {after}")
    if save:
        logger.info(f"Will save to: {save}")
    logger.info("Comparison not yet implemented (coming in Phase 8)")


if __name__ == '__main__':
    main()
