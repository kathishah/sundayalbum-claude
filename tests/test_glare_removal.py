"""Tests for single-shot glare removal."""

import numpy as np
import pytest

from src.glare.remover_single import (
    remove_glare_single,
    GlareResult,
    _feather_mask,
    _apply_intensity_correction,
    _apply_inpainting,
    _apply_contextual_fill,
    _blend_with_feathering,
    _match_boundary_colors,
)


class TestFeatherMask:
    """Tests for mask feathering."""

    def test_feather_mask_basic(self):
        """Test basic mask feathering."""
        # Create sharp binary mask
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0

        # Feather it
        feathered = _feather_mask(mask, radius=5)

        # Check properties
        assert feathered.shape == mask.shape
        assert feathered.dtype == np.float32
        assert 0.0 <= feathered.min() <= feathered.max() <= 1.0

        # Center should still be high (close to 1)
        assert feathered[50, 50] > 0.9

        # Edges should be intermediate (transition zone)
        assert 0.2 < feathered[30, 50] < 0.8

        # Outside should be low (close to 0)
        assert feathered[10, 10] < 0.1

    def test_feather_mask_zero_radius(self):
        """Test that zero radius returns original mask."""
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[20:30, 20:30] = 1.0

        feathered = _feather_mask(mask, radius=0)

        np.testing.assert_array_equal(feathered, mask)


class TestIntensityCorrection:
    """Tests for intensity correction method."""

    def test_intensity_correction_basic(self):
        """Test intensity correction on mild glare."""
        # Create test image
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Create mild glare region (slightly brighter)
        original = image.copy()
        image[40:60, 40:60] = 0.8  # Mild glare

        # Create mask and severity map
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.3  # Mild severity

        # Apply correction
        result, confidence = _apply_intensity_correction(image, mask, severity, original)

        # Check properties
        assert result.shape == image.shape
        assert confidence.shape == (100, 100)
        assert result.dtype == np.float32
        assert confidence.dtype == np.float32

        # Confidence should be high for mild glare
        assert confidence[50, 50] > 0.7

        # Glare region should be corrected (darker than input)
        assert result[50, 50, 0] < image[50, 50, 0]


class TestInpainting:
    """Tests for OpenCV inpainting method."""

    def test_inpainting_basic(self):
        """Test inpainting on moderate glare."""
        # Create test image with gradient
        image = np.zeros((100, 100, 3), dtype=np.float32)
        for i in range(100):
            image[i, :, :] = i / 100.0  # Vertical gradient

        # Create glare mask in the middle
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True

        # Apply inpainting
        result, confidence = _apply_inpainting(image, mask, radius=5)

        # Check properties
        assert result.shape == image.shape
        assert confidence.shape == (100, 100)
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

        # Confidence in glare region should be moderate
        assert 0.4 < confidence[50, 50] < 0.8

        # Non-glare regions should be unchanged
        np.testing.assert_array_almost_equal(result[10, 50], image[10, 50], decimal=2)

    def test_inpainting_preserves_non_glare(self):
        """Test that inpainting doesn't change non-glare regions."""
        image = np.random.rand(80, 80, 3).astype(np.float32)

        # Small glare region
        mask = np.zeros((80, 80), dtype=bool)
        mask[35:45, 35:45] = True

        result, _ = _apply_inpainting(image, mask, radius=3)

        # Non-glare pixels should be very similar
        non_glare_diff = np.abs(result[~mask] - image[~mask]).mean()
        assert non_glare_diff < 0.01


class TestContextualFill:
    """Tests for contextual fill method."""

    def test_contextual_fill_basic(self):
        """Test contextual fill on severe glare."""
        # Create test image
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.6
        original = image.copy()

        # Large severe glare region
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True

        # Apply contextual fill
        result, confidence = _apply_contextual_fill(image, mask, original)

        # Check properties
        assert result.shape == image.shape
        assert confidence.shape == (100, 100)
        assert result.dtype == np.float32

        # Confidence should be low for severe glare
        assert confidence[50, 50] < 0.5

        # Result should be in valid range
        assert 0.0 <= result.min() <= result.max() <= 1.0


class TestBlendWithFeathering:
    """Tests for feathered blending."""

    def test_blend_with_feathering(self):
        """Test blending original and corrected images."""
        original = np.ones((100, 100, 3), dtype=np.float32) * 0.3
        corrected = np.ones((100, 100, 3), dtype=np.float32) * 0.7

        # Feathered mask (0 to 1 gradient)
        mask = np.zeros((100, 100), dtype=np.float32)
        for i in range(100):
            mask[:, i] = i / 100.0  # Horizontal gradient

        result = _blend_with_feathering(original, corrected, mask)

        # Check properties
        assert result.shape == original.shape
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

        # Left edge should be mostly original (mask ~0)
        assert np.abs(result[50, 0, 0] - original[50, 0, 0]) < 0.05

        # Right edge should be mostly corrected (mask ~1)
        assert np.abs(result[50, 99, 0] - corrected[50, 99, 0]) < 0.05

        # Middle should be blend
        expected_middle = 0.3 * 0.5 + 0.7 * 0.5
        assert np.abs(result[50, 50, 0] - expected_middle) < 0.05


class TestMatchBoundaryColors:
    """Tests for boundary color matching."""

    def test_match_boundary_colors_basic(self):
        """Test color matching at boundaries."""
        # Create image with different color inside glare region
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5
        reference = image.copy()

        # Make glare region different
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[40:60, 40:60] = 1.0
        image[40:60, 40:60] = 0.8  # Brighter in glare region

        # Match colors
        result = _match_boundary_colors(image, mask, reference)

        # Check properties
        assert result.shape == image.shape
        assert result.dtype == np.float32
        assert 0.0 <= result.min() <= result.max() <= 1.0

        # Non-glare should be unchanged
        np.testing.assert_array_almost_equal(result[10, 10], image[10, 10], decimal=2)


class TestRemoveGlareSingle:
    """Integration tests for remove_glare_single."""

    def test_no_glare(self):
        """Test that images with no glare pass through unchanged."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.zeros((100, 100), dtype=np.uint8)
        severity = np.zeros((100, 100), dtype=np.float32)

        result = remove_glare_single(image, mask, severity)

        assert isinstance(result, GlareResult)
        assert result.image.shape == image.shape
        assert result.confidence_map.shape == (100, 100)
        assert result.method_used == {"none": 100.0}

        # Image should be identical (or very close)
        np.testing.assert_array_almost_equal(result.image, image, decimal=5)

        # Confidence should be perfect
        assert np.all(result.confidence_map == 1.0)

    def test_mild_glare_only(self):
        """Test glare removal with only mild glare."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Mild glare region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.3  # Mild

        result = remove_glare_single(image, mask, severity, inpaint_radius=5, feather_radius=3)

        assert isinstance(result, GlareResult)
        assert result.image.shape == image.shape
        assert result.method_used["intensity"] > 0  # Intensity correction used
        assert result.method_used["inpaint"] == 0  # Inpainting not used
        assert result.method_used["contextual"] == 0  # Contextual not used

    def test_moderate_glare_only(self):
        """Test glare removal with only moderate glare."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Moderate glare region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.55  # Moderate

        result = remove_glare_single(image, mask, severity, inpaint_radius=5)

        assert isinstance(result, GlareResult)
        assert result.method_used["intensity"] == 0  # Intensity not used
        assert result.method_used["inpaint"] > 0  # Inpainting used
        assert result.method_used["contextual"] == 0  # Contextual not used

    def test_severe_glare_only(self):
        """Test glare removal with only severe glare."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        # Severe glare region
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[30:70, 30:70] = 0.85  # Severe

        result = remove_glare_single(image, mask, severity, inpaint_radius=5)

        assert isinstance(result, GlareResult)
        assert result.method_used["intensity"] == 0  # Intensity not used
        assert result.method_used["inpaint"] == 0  # Inpainting not used
        assert result.method_used["contextual"] > 0  # Contextual used

    def test_mixed_severity_glare(self):
        """Test glare removal with multiple severity levels."""
        image = np.ones((120, 120, 3), dtype=np.float32) * 0.5

        # Create regions with different severity
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

        result = remove_glare_single(image, mask, severity, inpaint_radius=5)

        assert isinstance(result, GlareResult)

        # All three methods should be used
        assert result.method_used["intensity"] > 0
        assert result.method_used["inpaint"] > 0
        assert result.method_used["contextual"] > 0

        # Total should be 100%
        total = sum(result.method_used.values())
        assert 99.9 < total < 100.1

    def test_confidence_map_shape(self):
        """Test confidence map has correct shape and values."""
        image = np.ones((80, 80, 3), dtype=np.float32) * 0.5

        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[30:50, 30:50] = 255

        severity = np.zeros((80, 80), dtype=np.float32)
        severity[30:50, 30:50] = 0.5

        result = remove_glare_single(image, mask, severity)

        # Confidence map should have correct shape
        assert result.confidence_map.shape == (80, 80)
        assert result.confidence_map.dtype == np.float32

        # Values should be in [0, 1]
        assert 0.0 <= result.confidence_map.min() <= result.confidence_map.max() <= 1.0

        # Non-glare regions should have confidence = 1.0
        assert np.all(result.confidence_map[10, 10] == 1.0)

    def test_result_in_valid_range(self):
        """Test that result image values are in valid range."""
        image = np.random.rand(100, 100, 3).astype(np.float32)

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.6

        result = remove_glare_single(image, mask, severity)

        # Result should be in [0, 1]
        assert 0.0 <= result.image.min()
        assert result.image.max() <= 1.0

    def test_custom_inpaint_radius(self):
        """Test that custom inpaint radius is respected."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.5

        # Should not crash with different radii
        result1 = remove_glare_single(image, mask, severity, inpaint_radius=3)
        result2 = remove_glare_single(image, mask, severity, inpaint_radius=10)

        assert isinstance(result1, GlareResult)
        assert isinstance(result2, GlareResult)

    def test_custom_feather_radius(self):
        """Test that custom feather radius is respected."""
        image = np.ones((100, 100, 3), dtype=np.float32) * 0.5

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255

        severity = np.zeros((100, 100), dtype=np.float32)
        severity[40:60, 40:60] = 0.5

        # Should not crash with different feather radii
        result1 = remove_glare_single(image, mask, severity, feather_radius=2)
        result2 = remove_glare_single(image, mask, severity, feather_radius=10)

        assert isinstance(result1, GlareResult)
        assert isinstance(result2, GlareResult)


class TestGlareRemovalRealImages:
    """Integration tests with real test images (skipped if images unavailable)."""

    @pytest.fixture
    def test_images_available(self):
        """Check if test images are available."""
        from pathlib import Path
        test_images_dir = Path("test-images")
        return test_images_dir.exists() and any(test_images_dir.glob("*.HEIC"))

    def test_glare_removal_on_cave_image(self, test_images_available):
        """Test glare removal on cave image (print glare)."""
        if not test_images_available:
            pytest.skip("Test images not available")

        from src.preprocessing.loader import load_image
        from src.preprocessing.normalizer import normalize
        from src.glare.detector import detect_glare

        # Load and normalize
        image, _ = load_image("test-images/IMG_cave_normal.HEIC")
        norm_result = normalize(image, max_working_resolution=2000)

        # Detect glare
        glare_detection = detect_glare(norm_result.image)

        # Remove glare
        if glare_detection.total_glare_area_ratio > 0.001:
            result = remove_glare_single(
                norm_result.image,
                glare_detection.mask,
                glare_detection.severity_map
            )

            assert isinstance(result, GlareResult)
            assert result.image.shape == norm_result.image.shape
            assert 0.0 <= result.image.min() <= result.image.max() <= 1.0
            print(f"Cave image glare removal: {result.method_used}")

    def test_glare_removal_on_three_pics_image(self, test_images_available):
        """Test glare removal on three_pics image (sleeve glare)."""
        if not test_images_available:
            pytest.skip("Test images not available")

        from src.preprocessing.loader import load_image
        from src.preprocessing.normalizer import normalize
        from src.glare.detector import detect_glare

        # Load and normalize
        image, _ = load_image("test-images/IMG_three_pics_normal.HEIC")
        norm_result = normalize(image, max_working_resolution=2000)

        # Detect glare
        glare_detection = detect_glare(norm_result.image)

        # Remove glare
        if glare_detection.total_glare_area_ratio > 0.001:
            result = remove_glare_single(
                norm_result.image,
                glare_detection.mask,
                glare_detection.severity_map
            )

            assert isinstance(result, GlareResult)
            assert result.image.shape == norm_result.image.shape
            print(f"Three pics glare removal: {result.method_used}")
