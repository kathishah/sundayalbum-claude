#!/usr/bin/env python3
"""Validation script for glare detection on all test images.

Run this after test images are downloaded:
    python3 scripts/validate_glare_detection.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.loader import load_image
from src.glare.detector import detect_glare
from src.glare.confidence import compute_glare_confidence


def main():
    """Run glare detection validation on all HEIC test images."""
    test_images_dir = Path(__file__).parent.parent / "test-images"

    if not test_images_dir.exists():
        print("ERROR: Test images directory not found.")
        print("Run: bash scripts/fetch-test-images.sh")
        sys.exit(1)

    heic_files = sorted(test_images_dir.glob("*.HEIC"))

    if not heic_files:
        print("ERROR: No HEIC files found in test-images/")
        sys.exit(1)

    print(f"\nGlare Detection Validation on {len(heic_files)} HEIC images")
    print("=" * 100)

    results = []

    for heic_file in heic_files:
        print(f"\nProcessing: {heic_file.name}")

        # Load image
        try:
            img, metadata = load_image(str(heic_file))
            print(f"  Loaded: {metadata.original_size[0]}x{metadata.original_size[1]}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        # Detect glare
        try:
            detection = detect_glare(img)
            confidence = compute_glare_confidence(img, detection.mask)

            results.append(
                {
                    "filename": heic_file.name,
                    "glare_detected": detection.total_glare_area_ratio > 0.01,
                    "glare_type": detection.glare_type,
                    "area_ratio": detection.total_glare_area_ratio,
                    "confidence": confidence,
                    "num_regions": len(detection.regions),
                }
            )

            print(f"  Glare Type: {detection.glare_type}")
            print(f"  Area Ratio: {detection.total_glare_area_ratio:.4f} ({detection.total_glare_area_ratio*100:.2f}%)")
            print(f"  Regions: {len(detection.regions)}")
            print(f"  Confidence: {confidence:.4f}")

        except Exception as e:
            print(f"  ERROR detecting glare: {e}")
            continue

    # Print summary table
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Filename':<45} | {'Detected':<8} | {'Type':<7} | {'Area %':<7} | {'Conf':<5} | {'Regions':<7}"
    )
    print("-" * 100)

    for r in results:
        print(
            f"{r['filename']:<45} | {str(r['glare_detected']):<8} | {r['glare_type']:<7} | "
            f"{r['area_ratio']*100:>6.2f}% | {r['confidence']:>4.2f} | {r['num_regions']:>7}"
        )

    print("=" * 100)
    print(f"\nValidated {len(results)} images successfully.")


if __name__ == "__main__":
    main()
