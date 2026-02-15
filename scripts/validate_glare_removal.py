#!/usr/bin/env python3
"""Validation script for glare removal functionality.

This script tests the glare removal on synthetic images without requiring pytest.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.glare.remover_single import remove_glare_single, GlareResult
    print("✓ Successfully imported glare removal module")
except ImportError as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)


def test_no_glare():
    """Test with no glare."""
    print("\n[Test 1] No glare case...")
    image = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    mask = np.zeros((100, 100), dtype=np.uint8)
    severity = np.zeros((100, 100), dtype=np.float32)

    result = remove_glare_single(image, mask, severity)

    assert isinstance(result, GlareResult), "Result should be GlareResult"
    assert result.method_used == {"none": 100.0}, f"Expected no processing, got {result.method_used}"
    print(f"  ✓ Passed - method used: {result.method_used}")


def test_mild_glare():
    """Test with mild glare only."""
    print("\n[Test 2] Mild glare only...")
    image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    severity = np.zeros((100, 100), dtype=np.float32)
    severity[40:60, 40:60] = 0.3  # Mild

    result = remove_glare_single(image, mask, severity)

    assert result.method_used["intensity"] > 0, "Intensity correction should be used"
    assert result.method_used["inpaint"] == 0, "Inpainting should not be used"
    assert result.method_used["contextual"] == 0, "Contextual should not be used"
    print(f"  ✓ Passed - method breakdown: {result.method_used}")


def test_moderate_glare():
    """Test with moderate glare only."""
    print("\n[Test 3] Moderate glare only...")
    image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    severity = np.zeros((100, 100), dtype=np.float32)
    severity[40:60, 40:60] = 0.55  # Moderate

    result = remove_glare_single(image, mask, severity)

    assert result.method_used["intensity"] == 0, "Intensity should not be used"
    assert result.method_used["inpaint"] > 0, "Inpainting should be used"
    assert result.method_used["contextual"] == 0, "Contextual should not be used"
    print(f"  ✓ Passed - method breakdown: {result.method_used}")


def test_severe_glare():
    """Test with severe glare only."""
    print("\n[Test 4] Severe glare only...")
    image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[30:70, 30:70] = 255

    severity = np.zeros((100, 100), dtype=np.float32)
    severity[30:70, 30:70] = 0.85  # Severe

    result = remove_glare_single(image, mask, severity)

    assert result.method_used["intensity"] == 0, "Intensity should not be used"
    assert result.method_used["inpaint"] == 0, "Inpainting should not be used"
    assert result.method_used["contextual"] > 0, "Contextual should be used"
    print(f"  ✓ Passed - method breakdown: {result.method_used}")


def test_mixed_severity():
    """Test with mixed severity levels."""
    print("\n[Test 5] Mixed severity glare...")
    image = np.ones((120, 120, 3), dtype=np.float32) * 0.5

    mask = np.zeros((120, 120), dtype=np.uint8)
    severity = np.zeros((120, 120), dtype=np.float32)

    # Mild region
    mask[10:30, 10:30] = 255
    severity[10:30, 10:30] = 0.3

    # Moderate region
    mask[40:60, 40:60] = 255
    severity[40:60, 40:60] = 0.55

    # Severe region
    mask[80:100, 80:100] = 255
    severity[80:100, 80:100] = 0.85

    result = remove_glare_single(image, mask, severity)

    assert result.method_used["intensity"] > 0, "Intensity should be used"
    assert result.method_used["inpaint"] > 0, "Inpainting should be used"
    assert result.method_used["contextual"] > 0, "Contextual should be used"

    total = sum(result.method_used.values())
    assert 99.9 < total < 100.1, f"Total should be ~100%, got {total}"
    print(f"  ✓ Passed - method breakdown: {result.method_used}")


def test_confidence_map():
    """Test confidence map properties."""
    print("\n[Test 6] Confidence map validation...")
    image = np.ones((80, 80, 3), dtype=np.float32) * 0.5

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[30:50, 30:50] = 255

    severity = np.zeros((80, 80), dtype=np.float32)
    severity[30:50, 30:50] = 0.5

    result = remove_glare_single(image, mask, severity)

    assert result.confidence_map.shape == (80, 80), "Confidence map shape mismatch"
    assert 0.0 <= result.confidence_map.min() <= result.confidence_map.max() <= 1.0, \
        "Confidence values out of range"
    assert result.confidence_map[10, 10] == 1.0, "Non-glare region should have confidence 1.0"
    print(f"  ✓ Passed - confidence range: [{result.confidence_map.min():.3f}, {result.confidence_map.max():.3f}]")


def test_output_range():
    """Test that output is in valid range."""
    print("\n[Test 7] Output range validation...")
    image = np.random.rand(100, 100, 3).astype(np.float32)

    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255

    severity = np.zeros((100, 100), dtype=np.float32)
    severity[40:60, 40:60] = 0.6

    result = remove_glare_single(image, mask, severity)

    assert 0.0 <= result.image.min(), f"Min value {result.image.min()} < 0"
    assert result.image.max() <= 1.0, f"Max value {result.image.max()} > 1"
    print(f"  ✓ Passed - output range: [{result.image.min():.3f}, {result.image.max():.3f}]")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Glare Removal Validation Tests")
    print("=" * 60)

    tests = [
        test_no_glare,
        test_mild_glare,
        test_moderate_glare,
        test_severe_glare,
        test_mixed_severity,
        test_confidence_map,
        test_output_range,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
