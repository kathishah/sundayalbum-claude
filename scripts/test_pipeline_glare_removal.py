#!/usr/bin/env python3
"""End-to-end pipeline test with glare removal on synthetic image."""

import sys
import time
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.pipeline import Pipeline, PipelineConfig
    print("✓ Successfully imported pipeline module")
except ImportError as e:
    print(f"✗ Failed to import pipeline: {e}")
    sys.exit(1)


def create_test_image_with_glare():
    """Create a synthetic test image with glare region."""
    # Create 1000x1000 test image with gradient
    image = np.zeros((1000, 1000, 3), dtype=np.float32)

    # Add a gradient background
    for i in range(1000):
        for j in range(1000):
            image[i, j, 0] = i / 1000.0  # Red gradient vertical
            image[i, j, 1] = j / 1000.0  # Green gradient horizontal
            image[i, j, 2] = 0.5  # Constant blue

    # Add some "photos" (darker rectangles)
    image[200:400, 200:400] = [0.3, 0.2, 0.4]  # Photo 1
    image[200:400, 600:800] = [0.4, 0.3, 0.2]  # Photo 2
    image[600:800, 400:600] = [0.2, 0.4, 0.3]  # Photo 3

    # Add glare region (bright, desaturated)
    # Create a large glare region with varying severity
    glare_center_x, glare_center_y = 500, 300
    glare_radius = 150

    for i in range(1000):
        for j in range(1000):
            dist = np.sqrt((i - glare_center_y)**2 + (j - glare_center_x)**2)
            if dist < glare_radius:
                # Closer to center = more glare
                glare_strength = 1.0 - (dist / glare_radius)
                # Make it bright and desaturated
                glare_color = np.array([0.95, 0.95, 0.95], dtype=np.float32)
                image[i, j] = image[i, j] * (1.0 - glare_strength) + glare_color * glare_strength

    return image


def save_test_image(image, path):
    """Save test image as JPEG."""
    import cv2

    # Convert to uint8 BGR for OpenCV
    image_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), image_bgr)
    print(f"  Saved test image to {path}")


def main():
    """Run end-to-end pipeline test with glare removal."""
    print("=" * 70)
    print("Pipeline Test: Glare Removal on Synthetic Image")
    print("=" * 70)

    # Create output directories
    output_dir = Path("output")
    debug_dir = Path("debug")
    output_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    # Create test image
    print("\n[1] Creating synthetic test image with glare...")
    test_image = create_test_image_with_glare()
    test_image_path = output_dir / "test_image_with_glare.jpg"
    save_test_image(test_image, test_image_path)
    print(f"  ✓ Test image created: {test_image.shape}, range [{test_image.min():.3f}, {test_image.max():.3f}]")

    # Create pipeline
    print("\n[2] Initializing pipeline...")
    config = PipelineConfig()
    pipeline = Pipeline(config)
    print(f"  ✓ Pipeline initialized with config")

    # Process image
    print("\n[3] Processing image through pipeline...")
    start_time = time.time()

    try:
        result = pipeline.process(
            str(test_image_path),
            debug_output_dir=str(debug_dir / "test_glare_removal"),
        )
        processing_time = time.time() - start_time

        print(f"  ✓ Processing complete in {processing_time:.3f}s")
        print(f"  ✓ Steps completed: {', '.join(result.steps_completed)}")

    except Exception as e:
        print(f"  ✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print results
    print("\n[4] Results:")
    print(f"  Processing time: {result.processing_time:.3f}s")
    print(f"  Steps completed: {len(result.steps_completed)}")
    print(f"  Output images: {len(result.output_images)}")

    if result.glare_detection:
        print(f"\n  Glare Detection:")
        print(f"    Type: {result.glare_detection.glare_type}")
        print(f"    Area ratio: {result.glare_detection.total_glare_area_ratio:.4f}")
        print(f"    Regions: {len(result.glare_detection.regions)}")
        print(f"    Confidence: {result.glare_confidence:.4f}")

    if result.glare_removal:
        print(f"\n  Glare Removal:")
        print(f"    Methods used: {result.glare_removal.method_used}")
        avg_confidence = result.glare_removal.confidence_map.mean()
        print(f"    Avg confidence: {avg_confidence:.4f}")

    # Print step times
    print(f"\n  Step Times:")
    for step, step_time in pipeline.step_times.items():
        pct = (step_time / result.processing_time) * 100
        print(f"    {step:20s}: {step_time:6.3f}s ({pct:5.1f}%)")

    # Save final output
    if result.output_images:
        output_path = output_dir / "test_glare_removal_result.jpg"
        save_test_image(result.output_images[0], output_path)
        print(f"\n  ✓ Final output saved to {output_path}")

    # List debug files
    debug_files = sorted((debug_dir / "test_glare_removal").glob("*"))
    print(f"\n  Debug files created ({len(debug_files)}):")
    for f in debug_files:
        print(f"    - {f.name}")

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
