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
            for i, output_image in enumerate(result.output_images, 1):
                output_filename = f"{input_file.stem}_processed_{i:02d}.jpg"
                output_file_path = output_path / output_filename

                save_debug_image(
                    output_image,
                    output_file_path,
                    f"Final output {i}",
                    quality=config.jpeg_quality
                )

                logger.info(f"Saved output: {output_file_path}")

            results.append(result)

            # Print summary
            logger.info(f"\nProcessing Summary:")
            logger.info(f"  Format: {result.metadata.format}")
            logger.info(f"  Original size: {result.metadata.original_size[0]}x{result.metadata.original_size[1]}")
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
