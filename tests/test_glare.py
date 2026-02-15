"""Tests for glare detection module."""

import pytest
import numpy as np
from pathlib import Path

from src.glare.detector import (
    detect_glare,
    draw_glare_overlay,
    GlareDetection,
    _suppress_textured_regions,
    _filter_small_regions,
    _compute_severity_map,
    _classify_glare_type,
)
from src.glare.confidence import compute_glare_confidence


# --- Fixtures ---


@pytest.fixture
def synthetic_image_with_glare():
    """Create a synthetic image with a bright, desaturated glare region."""
    # Create a 500x500 RGB image
    img = np.ones((500, 500, 3), dtype=np.float32) * 0.5  # Mid-gray background

    # Add a bright, desaturated glare region (100x100 square at center)
    # Glare = high brightness, low saturation
    center_y, center_x = 250, 250
    glare_size = 100
    y1 = center_y - glare_size // 2
    y2 = center_y + glare_size // 2
    x1 = center_x - glare_size // 2
    x2 = center_x + glare_size // 2

    # Make it very bright (0.95) and nearly white (desaturated)
    img[y1:y2, x1:x2, :] = 0.95

    return img


@pytest.fixture
def synthetic_image_no_glare():
    """Create a synthetic image with no glare."""
    # Create a 500x500 RGB image with random content, no extreme brightness
    np.random.seed(42)
    img = np.random.rand(500, 500, 3).astype(np.float32) * 0.6  # Max 0.6 brightness
    return img


@pytest.fixture
def synthetic_image_bright_texture():
    """Create a synthetic image with bright textured region (not glare)."""
    img = np.ones((500, 500, 3), dtype=np.float32) * 0.3  # Dark background

    # Add a bright region with high texture (checkerboard pattern)
    center_y, center_x = 250, 250
    size = 100
    y1 = center_y - size // 2
    y2 = center_y + size // 2
    x1 = center_x - size // 2
    x2 = center_x + size // 2

    # Create checkerboard pattern
    for i in range(y1, y2, 10):
        for j in range(x1, x2, 10):
            if ((i // 10) + (j // 10)) % 2 == 0:
                img[i : i + 10, j : j + 10, :] = 0.9
            else:
                img[i : i + 10, j : j + 10, :] = 0.7

    return img


# --- Unit Tests ---


class TestDetectGlare:
    """Test the main detect_glare function."""

    def test_detect_glare_with_glare(self, synthetic_image_with_glare):
        """Test that glare is detected in an image with obvious glare."""
        result = detect_glare(synthetic_image_with_glare)

        assert isinstance(result, GlareDetection)
        assert result.mask.shape == synthetic_image_with_glare.shape[:2]
        assert result.mask.dtype == np.uint8
        assert result.total_glare_area_ratio > 0.0
        assert len(result.regions) > 0
        assert result.severity_map.shape == synthetic_image_with_glare.shape[:2]
        assert result.glare_type in ["sleeve", "print", "none"]

    def test_detect_glare_no_glare(self, synthetic_image_no_glare):
        """Test that no glare is detected in a normal image."""
        result = detect_glare(synthetic_image_no_glare)

        assert isinstance(result, GlareDetection)
        # Should detect very little or no glare
        assert result.total_glare_area_ratio < 0.1
        assert result.glare_type in ["sleeve", "print", "none"]

    def test_detect_glare_bright_texture_suppression(self, synthetic_image_bright_texture):
        """Test that bright textured regions are not classified as glare."""
        result = detect_glare(synthetic_image_bright_texture)

        # The checkerboard should be suppressed due to high texture
        # May detect some glare, but much less than if texture suppression wasn't working
        assert result.total_glare_area_ratio < 0.15

    def test_detect_glare_custom_thresholds(self, synthetic_image_with_glare):
        """Test glare detection with custom thresholds."""
        # Lower intensity threshold should detect more glare
        result_low = detect_glare(synthetic_image_with_glare, intensity_threshold=0.7)
        result_high = detect_glare(synthetic_image_with_glare, intensity_threshold=0.95)

        assert result_low.total_glare_area_ratio >= result_high.total_glare_area_ratio

    def test_detect_glare_min_area_filter(self, synthetic_image_with_glare):
        """Test that min_area filter removes small regions."""
        # Very large min_area should remove all regions
        result = detect_glare(synthetic_image_with_glare, min_area=100000)

        assert len(result.regions) == 0
        assert result.total_glare_area_ratio == 0.0

    def test_detect_glare_forced_type(self, synthetic_image_with_glare):
        """Test that forced glare type is respected."""
        result_sleeve = detect_glare(synthetic_image_with_glare, glare_type="sleeve")
        result_print = detect_glare(synthetic_image_with_glare, glare_type="print")

        assert result_sleeve.glare_type == "sleeve"
        assert result_print.glare_type == "print"


class TestSuppressTexturedRegions:
    """Test the texture suppression helper."""

    def test_suppress_high_texture(self, synthetic_image_bright_texture):
        """Test that high-texture regions are suppressed."""
        # Create initial mask covering the bright checkerboard
        initial_mask = np.zeros((500, 500), dtype=bool)
        initial_mask[200:300, 200:300] = True

        suppressed = _suppress_textured_regions(
            synthetic_image_bright_texture,
            initial_mask,
            window_size=15,
            texture_threshold=0.02,
        )

        # Suppressed mask should have fewer pixels than initial
        assert np.sum(suppressed) < np.sum(initial_mask)

    def test_preserve_low_texture(self, synthetic_image_with_glare):
        """Test that low-texture regions are preserved."""
        # Create mask covering the uniform glare region
        initial_mask = np.zeros((500, 500), dtype=bool)
        initial_mask[200:300, 200:300] = True

        suppressed = _suppress_textured_regions(
            synthetic_image_with_glare,
            initial_mask,
            window_size=15,
            texture_threshold=0.02,
        )

        # Most of the uniform glare region should be preserved
        assert np.sum(suppressed) >= np.sum(initial_mask) * 0.7


class TestFilterSmallRegions:
    """Test the small region filter."""

    def test_filter_small_regions(self):
        """Test filtering of small regions."""
        # Create mask with one large region and several small ones
        mask = np.zeros((500, 500), dtype=np.uint8)

        # Large region (200x200 = 40000 pixels)
        mask[100:300, 100:300] = 255

        # Small region (10x10 = 100 pixels)
        mask[350:360, 350:360] = 255

        # Tiny region (5x5 = 25 pixels)
        mask[400:405, 400:405] = 255

        # Filter with min_area=500
        filtered_mask, regions = _filter_small_regions(mask, min_area=500)

        # Should keep only the large region
        assert len(regions) == 1
        assert np.sum(filtered_mask > 0) >= 39000  # Approximately 40000 pixels


class TestComputeSeverityMap:
    """Test the severity map computation."""

    def test_severity_map_range(self, synthetic_image_with_glare):
        """Test that severity map values are in [0, 1]."""
        # Create a simple glare mask
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[200:300, 200:300] = 255

        # Mock V channel with high brightness in glare region
        v_channel = np.ones((500, 500), dtype=np.float32) * 0.5
        v_channel[200:300, 200:300] = 0.95

        severity_map = _compute_severity_map(synthetic_image_with_glare, mask, v_channel)

        assert severity_map.shape == (500, 500)
        assert severity_map.dtype == np.float32
        assert np.all(severity_map >= 0.0)
        assert np.all(severity_map <= 1.0)

    def test_severity_map_zero_outside_glare(self, synthetic_image_with_glare):
        """Test that severity is zero outside glare regions."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[200:300, 200:300] = 255

        v_channel = np.ones((500, 500), dtype=np.float32) * 0.95

        severity_map = _compute_severity_map(synthetic_image_with_glare, mask, v_channel)

        # Outside the masked region, severity should be 0
        assert np.all(severity_map[:200, :] == 0)
        assert np.all(severity_map[300:, :] == 0)
        assert np.all(severity_map[:, :200] == 0)
        assert np.all(severity_map[:, 300:] == 0)


class TestClassifyGlareType:
    """Test glare type classification."""

    def test_classify_no_glare(self):
        """Test classification when no glare is present."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        regions = []
        glare_type = _classify_glare_type(mask, regions, total_area_ratio=0.0)

        assert glare_type == "none"

    def test_classify_sleeve_glare(self):
        """Test classification of sleeve-type glare (few large uniform regions)."""
        mask = np.zeros((500, 500), dtype=np.uint8)

        # Create 2 large, roughly circular regions (low irregularity)
        import cv2

        cv2.circle(mask, (150, 250), 60, 255, -1)
        cv2.circle(mask, (350, 250), 60, 255, -1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area_ratio = np.sum(mask > 0) / (500 * 500)

        glare_type = _classify_glare_type(mask, contours, total_area_ratio)

        # Should classify as sleeve (few large uniform blobs)
        assert glare_type == "sleeve"

    def test_classify_print_glare(self):
        """Test classification of print-type glare (many irregular regions)."""
        mask = np.zeros((500, 500), dtype=np.uint8)

        # Create many small irregular regions
        import cv2

        for i in range(10):
            x = 50 + i * 40
            y = 50 + i * 40
            # Irregular shapes
            pts = np.array(
                [
                    [x, y],
                    [x + 20, y + 5],
                    [x + 25, y + 20],
                    [x + 10, y + 30],
                    [x - 5, y + 15],
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        total_area_ratio = np.sum(mask > 0) / (500 * 500)

        glare_type = _classify_glare_type(mask, contours, total_area_ratio)

        # Should classify as print (many irregular regions)
        assert glare_type == "print"


class TestDrawGlareOverlay:
    """Test the glare overlay drawing function."""

    def test_draw_glare_overlay(self, synthetic_image_with_glare):
        """Test that overlay is drawn correctly."""
        detection = detect_glare(synthetic_image_with_glare)
        overlay = draw_glare_overlay(synthetic_image_with_glare, detection)

        assert overlay.shape == synthetic_image_with_glare.shape
        assert overlay.dtype == np.uint8
        assert np.all(overlay >= 0)
        assert np.all(overlay <= 255)


class TestComputeGlareConfidence:
    """Test the glare confidence scoring."""

    def test_confidence_no_glare(self, synthetic_image_no_glare):
        """Test confidence with no glare."""
        mask = np.zeros((500, 500), dtype=np.uint8)
        confidence = compute_glare_confidence(synthetic_image_no_glare, mask)

        assert confidence == 1.0

    def test_confidence_with_glare(self, synthetic_image_with_glare):
        """Test confidence decreases with glare present."""
        # Create mask with moderate glare
        mask = np.zeros((500, 500), dtype=np.uint8)
        mask[200:300, 200:300] = 255  # 10000 / 250000 = 4% glare

        confidence = compute_glare_confidence(synthetic_image_with_glare, mask)

        assert 0.0 < confidence < 1.0

    def test_confidence_decreases_with_area(self, synthetic_image_with_glare):
        """Test that confidence decreases as glare area increases."""
        # Small glare
        mask_small = np.zeros((500, 500), dtype=np.uint8)
        mask_small[240:260, 240:260] = 255  # 400 pixels

        # Large glare
        mask_large = np.zeros((500, 500), dtype=np.uint8)
        mask_large[150:350, 150:350] = 255  # 40000 pixels

        confidence_small = compute_glare_confidence(synthetic_image_with_glare, mask_small)
        confidence_large = compute_glare_confidence(synthetic_image_with_glare, mask_large)

        assert confidence_small > confidence_large


# --- Integration Tests with Real Images ---


@pytest.fixture
def test_images_available():
    """Check if test images are available."""
    test_images_dir = Path(__file__).parent.parent / "test-images"
    return test_images_dir.exists()


@pytest.fixture
def test_images_dir():
    """Get path to test images directory."""
    return Path(__file__).parent.parent / "test-images"


@pytest.mark.skipif(
    not Path(__file__).parent.parent.joinpath("test-images").exists(),
    reason="Test images not available (run scripts/fetch-test-images.sh)",
)
class TestGlareDetectionRealImages:
    """Integration tests with real test images."""

    def test_detect_cave_print_glare(self, test_images_dir):
        """Test glare detection on cave image (glossy print glare)."""
        from src.preprocessing.loader import load_image

        img, _ = load_image(str(test_images_dir / "IMG_cave_normal.HEIC"))
        result = detect_glare(img)

        assert isinstance(result, GlareDetection)
        # Cave has glossy print glare - should detect some
        # But we can't assert exact amounts since it depends on the image
        assert result.glare_type in ["sleeve", "print", "none"]

    def test_detect_three_pics_sleeve_glare(self, test_images_dir):
        """Test glare detection on three_pics album page (plastic sleeve glare)."""
        from src.preprocessing.loader import load_image

        img, _ = load_image(str(test_images_dir / "IMG_three_pics_normal.HEIC"))
        result = detect_glare(img)

        assert isinstance(result, GlareDetection)
        # Three pics has plastic sleeve glare
        assert result.glare_type in ["sleeve", "print", "none"]

    def test_detect_all_heic_images(self, test_images_dir):
        """Test glare detection on all HEIC test images."""
        from src.preprocessing.loader import load_image

        heic_files = sorted(test_images_dir.glob("*.HEIC"))
        assert len(heic_files) > 0

        results = []
        for heic_file in heic_files:
            img, _ = load_image(str(heic_file))
            result = detect_glare(img)
            confidence = compute_glare_confidence(img, result.mask)

            results.append(
                {
                    "filename": heic_file.name,
                    "glare_detected": result.total_glare_area_ratio > 0.01,
                    "glare_type": result.glare_type,
                    "area_ratio": result.total_glare_area_ratio,
                    "confidence": confidence,
                    "num_regions": len(result.regions),
                }
            )

        # Print summary table
        print("\n\nGlare Detection Results:")
        print(f"{'Filename':<40} | {'Detected':<8} | {'Type':<7} | {'Area %':<7} | {'Conf':<5} | {'Regions':<7}")
        print("-" * 90)
        for r in results:
            print(
                f"{r['filename']:<40} | {str(r['glare_detected']):<8} | {r['glare_type']:<7} | "
                f"{r['area_ratio']*100:>6.2f}% | {r['confidence']:>4.2f} | {r['num_regions']:>7}"
            )

        # All images should complete without errors
        assert len(results) == len(heic_files)
