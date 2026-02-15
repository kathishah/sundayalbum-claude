#!/usr/bin/env python3
"""Test the full pipeline with a synthetic image to verify integration."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import Pipeline, PipelineConfig


def create_synthetic_test_image() -> str:
    """Create a synthetic test image with glare."""
    # Create a 1000x1000 RGB image
    img_array = np.ones((1000, 1000, 3), dtype=np.uint8) * 128  # Gray background

    # Add some photo content (colored squares)
    img_array[200:400, 200:400] = [100, 150, 200]  # Blue-ish square
    img_array[200:400, 600:800] = [200, 100, 100]  # Red-ish square
    img_array[600:800, 400:600] = [100, 200, 100]  # Green-ish square

    # Add bright glare region (simulated)
    # Top-right corner - very bright, nearly white
    img_array[100:300, 700:900] = 250

    # Convert to PIL Image
    img_pil = Image.fromarray(img_array, mode='RGB')

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img_pil.save(f.name, quality=95)
        return f.name


def main():
    """Run pipeline on synthetic test image."""
    print("Creating synthetic test image...")
    test_image_path = create_synthetic_test_image()
    print(f"Created: {test_image_path}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "synthetic_test"
    debug_dir = Path(__file__).parent.parent / "debug" / "synthetic_test"

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning pipeline...")
    print(f"  Output: {output_dir}")
    print(f"  Debug: {debug_dir}")

    # Create pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)

    # Process
    try:
        result = pipeline.process(
            test_image_path,
            debug_output_dir=str(debug_dir),
            steps_filter=None,  # Run all steps
        )

        print("\n" + "=" * 80)
        print("PIPELINE RESULT")
        print("=" * 80)
        print(f"Steps completed: {', '.join(result.steps_completed)}")
        print(f"Total time: {result.processing_time:.3f}s")

        for step_name, step_time in pipeline.step_times.items():
            print(f"  {step_name}: {step_time:.3f}s")

        if result.page_detection:
            print(f"\nPage detection:")
            print(f"  Full frame: {result.page_detection.is_full_frame}")
            print(f"  Confidence: {result.page_detection.confidence:.3f}")

        if result.glare_detection:
            print(f"\nGlare detection:")
            print(f"  Type: {result.glare_detection.glare_type}")
            print(f"  Area ratio: {result.glare_detection.total_glare_area_ratio:.4f}")
            print(f"  Regions: {len(result.glare_detection.regions)}")
            print(f"  Confidence: {result.glare_confidence:.3f}")

        print("\n" + "=" * 80)
        print(f"\nDebug outputs saved to: {debug_dir}")
        print("Check these files:")
        debug_files = sorted(debug_dir.glob("*"))
        for f in debug_files:
            print(f"  - {f.name}")

        print("\n✓ Pipeline test completed successfully!")

    except Exception as e:
        print(f"\n✗ Pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temp file
        Path(test_image_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
