#!/usr/bin/env python3
"""Run Phase 4 validation on all test images.

This script processes all HEIC test images through the full pipeline including
glare detection and removal, then reports which images had glare and the results.
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.pipeline import Pipeline, PipelineConfig
    print("✓ Successfully imported pipeline module")
except ImportError as e:
    print(f"✗ Failed to import pipeline: {e}")
    sys.exit(1)


def main():
    """Process all HEIC test images and report glare detection/removal results."""
    print("=" * 80)
    print("Phase 4 Validation: Glare Detection & Removal on Real Test Images")
    print("=" * 80)

    # Check for test images
    test_images_dir = Path("test-images")
    if not test_images_dir.exists():
        print("\n✗ Test images directory not found!")
        print("  Run: bash scripts/fetch-test-images.sh")
        return 1

    # Get all HEIC files
    heic_files = sorted(test_images_dir.glob("*.HEIC"))
    if not heic_files:
        print("\n✗ No HEIC test images found!")
        return 1

    print(f"\n✓ Found {len(heic_files)} HEIC test images")

    # Create output directories
    output_dir = Path("output")
    debug_dir = Path("debug")
    output_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    # Initialize pipeline
    config = PipelineConfig()
    pipeline = Pipeline(config)

    # Process each image
    results = []
    total_start = time.time()

    for i, image_path in enumerate(heic_files, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(heic_files)}] Processing: {image_path.name}")
        print(f"{'='*80}")

        image_debug_dir = debug_dir / image_path.stem

        try:
            start = time.time()
            result = pipeline.process(
                str(image_path),
                debug_output_dir=str(image_debug_dir),
            )
            elapsed = time.time() - start

            # Store results
            results.append({
                'filename': image_path.name,
                'result': result,
                'time': elapsed,
            })

            # Print immediate results
            print(f"\n✓ Processing complete in {elapsed:.2f}s")
            print(f"  Steps: {', '.join(result.steps_completed)}")

            if result.glare_detection:
                gd = result.glare_detection
                print(f"\n  Glare Detection:")
                print(f"    Type: {gd.glare_type}")
                print(f"    Area ratio: {gd.total_glare_area_ratio:.4f} ({gd.total_glare_area_ratio*100:.2f}%)")
                print(f"    Regions: {len(gd.regions)}")
                print(f"    Confidence: {result.glare_confidence:.4f}")

                if result.glare_removal and gd.total_glare_area_ratio > 0.001:
                    gr = result.glare_removal
                    print(f"\n  Glare Removal:")
                    print(f"    Methods used:")
                    for method, pct in gr.method_used.items():
                        if pct > 0:
                            print(f"      - {method}: {pct:.1f}%")
                    avg_conf = gr.confidence_map.mean()
                    print(f"    Avg confidence: {avg_conf:.4f}")
                else:
                    print(f"\n  Glare Removal: Skipped (no significant glare)")
            else:
                print(f"\n  No glare detection result")

            print(f"\n  Debug outputs saved to: {image_debug_dir}/")

        except Exception as e:
            print(f"\n✗ Processing failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'filename': image_path.name,
                'result': None,
                'error': str(e),
                'time': 0,
            })

    total_elapsed = time.time() - total_start

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Glare Detection & Removal Results")
    print("=" * 80)

    # Table header
    print(f"\n{'Image':<45} {'Glare?':<8} {'Type':<10} {'Area %':<10} {'Removal Methods':<30}")
    print("-" * 103)

    images_with_glare = []
    images_without_glare = []

    for r in results:
        if r['result'] is None:
            print(f"{r['filename']:<45} {'ERROR':<8}")
            continue

        result = r['result']
        gd = result.glare_detection

        if gd and gd.total_glare_area_ratio > 0.001:
            # Has glare
            images_with_glare.append(r['filename'])
            area_pct = f"{gd.total_glare_area_ratio*100:.2f}%"

            # Get removal methods
            if result.glare_removal:
                methods = ", ".join([
                    f"{m}({v:.0f}%)"
                    for m, v in result.glare_removal.method_used.items()
                    if v > 0
                ])
            else:
                methods = "not applied"

            print(f"{r['filename']:<45} {'YES':<8} {gd.glare_type:<10} {area_pct:<10} {methods:<30}")
        else:
            # No glare
            images_without_glare.append(r['filename'])
            print(f"{r['filename']:<45} {'NO':<8} {'-':<10} {'-':<10} {'-':<30}")

    # Summary stats
    print("\n" + "=" * 80)
    print(f"Images with glare: {len(images_with_glare)}/{len(results)}")
    if images_with_glare:
        for name in images_with_glare:
            print(f"  • {name}")

    print(f"\nImages without glare: {len(images_without_glare)}/{len(results)}")
    if images_without_glare:
        for name in images_without_glare:
            print(f"  • {name}")

    print(f"\nTotal processing time: {total_elapsed:.2f}s")
    print(f"Average time per image: {total_elapsed/len(results):.2f}s")

    # Glare type breakdown
    glare_types = {}
    for r in results:
        if r['result'] and r['result'].glare_detection:
            gtype = r['result'].glare_detection.glare_type
            glare_types[gtype] = glare_types.get(gtype, 0) + 1

    if glare_types:
        print(f"\nGlare type distribution:")
        for gtype, count in sorted(glare_types.items()):
            print(f"  {gtype}: {count}")

    print("\n" + "=" * 80)
    print("Validation complete! Check debug/ directory for visual results.")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
