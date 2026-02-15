"""Tests for page detection and perspective correction."""

import numpy as np
import pytest
from pathlib import Path

from src.page_detection.detector import (
    PageDetection,
    detect_page,
    draw_page_detection,
    _order_corners,
)
from src.page_detection.perspective import correct_perspective, _compute_output_dimensions


# Test data directory
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test-images"


@pytest.fixture
def test_images_exist():
    """Check if test images directory exists and has files."""
    if not TEST_IMAGES_DIR.exists():
        pytest.skip(f"Test images directory not found: {TEST_IMAGES_DIR}")

    heic_files = list(TEST_IMAGES_DIR.glob("*.HEIC")) + list(TEST_IMAGES_DIR.glob("*.heic"))
    if not heic_files:
        pytest.skip("No HEIC files found in test-images directory")

    return heic_files


def _create_synthetic_page_image(
    width: int = 800,
    height: int = 600,
    page_margin: int = 80,
    bg_color: float = 0.3,
    page_color: float = 0.85,
) -> np.ndarray:
    """Create a synthetic image with a lighter rectangle (page) on a dark background.

    Args:
        width: Image width.
        height: Image height.
        page_margin: Margin around the page rectangle.
        bg_color: Background brightness [0, 1].
        page_color: Page brightness [0, 1].

    Returns:
        Float32 RGB image [0, 1] with shape (height, width, 3).
    """
    image = np.full((height, width, 3), bg_color, dtype=np.float32)

    # Draw a lighter rectangle representing the page
    y1, y2 = page_margin, height - page_margin
    x1, x2 = page_margin, width - page_margin
    image[y1:y2, x1:x2] = page_color

    return image


def _create_synthetic_full_frame_image(
    width: int = 800,
    height: int = 600,
) -> np.ndarray:
    """Create a synthetic image with no clear page boundary (content fills frame).

    Returns:
        Float32 RGB image [0, 1] with shape (height, width, 3).
    """
    # Smooth gradient — no clear boundary to detect
    rng = np.random.RandomState(42)
    image = rng.uniform(0.3, 0.7, (height, width, 3)).astype(np.float32)

    # Apply heavy blur to remove edges
    import cv2
    img_uint8 = (image * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (51, 51), 0)
    return blurred.astype(np.float32) / 255.0


# --- Unit tests with synthetic images ---


class TestOrderCorners:
    """Test corner ordering utility."""

    def test_already_ordered(self) -> None:
        pts = np.array([[10, 10], [90, 10], [90, 90], [10, 90]], dtype=np.float32)
        ordered = _order_corners(pts)
        np.testing.assert_array_almost_equal(ordered[0], [10, 10])  # TL
        np.testing.assert_array_almost_equal(ordered[1], [90, 10])  # TR
        np.testing.assert_array_almost_equal(ordered[2], [90, 90])  # BR
        np.testing.assert_array_almost_equal(ordered[3], [10, 90])  # BL

    def test_shuffled_corners(self) -> None:
        pts = np.array([[90, 90], [10, 10], [10, 90], [90, 10]], dtype=np.float32)
        ordered = _order_corners(pts)
        np.testing.assert_array_almost_equal(ordered[0], [10, 10])  # TL
        np.testing.assert_array_almost_equal(ordered[1], [90, 10])  # TR
        np.testing.assert_array_almost_equal(ordered[2], [90, 90])  # BR
        np.testing.assert_array_almost_equal(ordered[3], [10, 90])  # BL


class TestComputeOutputDimensions:
    """Test output dimension computation from corners."""

    def test_rectangle(self) -> None:
        corners = np.array([[0, 0], [100, 0], [100, 50], [0, 50]], dtype=np.float32)
        w, h = _compute_output_dimensions(corners)
        assert w == 100
        assert h == 50

    def test_slightly_skewed(self) -> None:
        corners = np.array([[5, 0], [105, 5], [100, 55], [0, 50]], dtype=np.float32)
        w, h = _compute_output_dimensions(corners)
        # Should use max of opposite edges
        assert w > 95
        assert h > 45


class TestDetectPage:
    """Test page detection on synthetic images."""

    def test_clear_page_boundary(self) -> None:
        """A high-contrast rectangle on a dark background should be detected."""
        image = _create_synthetic_page_image(
            width=800, height=600, page_margin=80, bg_color=0.1, page_color=0.9
        )
        result = detect_page(image, min_area_ratio=0.1)

        assert isinstance(result, PageDetection)
        assert result.corners.shape == (4, 2)
        # Should detect the page, not go full-frame
        assert not result.is_full_frame, "Should detect the page boundary"
        assert result.confidence > 0.0

    def test_full_frame_no_boundary(self) -> None:
        """An image with no clear boundary should return full-frame."""
        image = _create_synthetic_full_frame_image(width=800, height=600)
        result = detect_page(image, min_area_ratio=0.3)

        assert isinstance(result, PageDetection)
        assert result.corners.shape == (4, 2)
        # With no clear boundary, should fall back to full frame
        assert result.is_full_frame

    def test_returns_four_corners(self) -> None:
        image = _create_synthetic_page_image()
        result = detect_page(image, min_area_ratio=0.1)
        assert result.corners.shape == (4, 2)

    def test_confidence_range(self) -> None:
        image = _create_synthetic_page_image()
        result = detect_page(image, min_area_ratio=0.1)
        assert 0.0 <= result.confidence <= 1.0


class TestCorrectPerspective:
    """Test perspective correction."""

    def test_identity_transform(self) -> None:
        """Corners matching the full image should produce similar output."""
        image = np.random.rand(100, 200, 3).astype(np.float32)
        corners = np.array([
            [0, 0], [199, 0], [199, 99], [0, 99]
        ], dtype=np.float32)

        result = correct_perspective(image, corners)
        assert result.shape[2] == 3
        assert result.dtype == np.float32
        # Output size should be close to input
        assert abs(result.shape[0] - 100) <= 2
        assert abs(result.shape[1] - 200) <= 2

    def test_crop_region(self) -> None:
        """Corners within the image should produce a cropped/warped result."""
        image = np.random.rand(200, 300, 3).astype(np.float32)
        corners = np.array([
            [50, 50], [250, 50], [250, 150], [50, 150]
        ], dtype=np.float32)

        result = correct_perspective(image, corners)
        assert result.dtype == np.float32
        assert result.shape[2] == 3
        # Output should be approximately 200x100
        assert abs(result.shape[1] - 200) <= 2
        assert abs(result.shape[0] - 100) <= 2

    def test_value_range_preserved(self) -> None:
        """Output should remain in [0, 1] range."""
        image = np.random.rand(100, 100, 3).astype(np.float32)
        corners = np.array([
            [10, 10], [90, 10], [90, 90], [10, 90]
        ], dtype=np.float32)

        result = correct_perspective(image, corners)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


class TestDrawPageDetection:
    """Test debug overlay drawing."""

    def test_overlay_same_size(self) -> None:
        image = np.random.rand(100, 200, 3).astype(np.float32)
        detection = PageDetection(
            corners=np.array([[10, 10], [190, 10], [190, 90], [10, 90]], dtype=np.float32),
            confidence=0.8,
            is_full_frame=False,
        )
        result = draw_page_detection(image, detection)
        assert result.shape == image.shape
        assert result.dtype == np.float32

    def test_overlay_full_frame(self) -> None:
        image = np.random.rand(100, 200, 3).astype(np.float32)
        detection = PageDetection(
            corners=np.array([[0, 0], [199, 0], [199, 99], [0, 99]], dtype=np.float32),
            confidence=0.0,
            is_full_frame=True,
        )
        result = draw_page_detection(image, detection)
        assert result.shape == image.shape


# --- Integration tests with real test images ---


def test_detect_page_three_pics(test_images_exist):
    """Test page detection on album page with 3 photos (should find album page boundary)."""
    heic_files = test_images_exist
    three_pics = [f for f in heic_files if "three_pics" in f.name]

    if not three_pics:
        pytest.skip("IMG_three_pics_normal.HEIC not found")

    from src.preprocessing.loader import load_image
    from src.preprocessing.normalizer import normalize

    image, _ = load_image(str(three_pics[0]))
    norm = normalize(image, max_working_resolution=2000)

    result = detect_page(norm.image)

    assert isinstance(result, PageDetection)
    assert result.corners.shape == (4, 2)
    print(f"three_pics: confidence={result.confidence:.3f}, full_frame={result.is_full_frame}")


def test_detect_page_cave(test_images_exist):
    """Test page detection on individual print (cave — should find print boundary or go full-frame)."""
    heic_files = test_images_exist
    cave_files = [f for f in heic_files if "cave" in f.name]

    if not cave_files:
        pytest.skip("IMG_cave_normal.HEIC not found")

    from src.preprocessing.loader import load_image
    from src.preprocessing.normalizer import normalize

    image, _ = load_image(str(cave_files[0]))
    norm = normalize(image, max_working_resolution=2000)

    result = detect_page(norm.image)

    assert isinstance(result, PageDetection)
    assert result.corners.shape == (4, 2)
    print(f"cave: confidence={result.confidence:.3f}, full_frame={result.is_full_frame}")


def test_detect_all_heic_images(test_images_exist):
    """Test page detection on all HEIC test images — print results summary."""
    heic_files = test_images_exist

    from src.preprocessing.loader import load_image
    from src.preprocessing.normalizer import normalize

    print(f"\n{'filename':<50} {'confidence':>10} {'full_frame':>12}")
    print("-" * 75)

    for heic_path in sorted(heic_files):
        image, _ = load_image(str(heic_path))
        norm = normalize(image, max_working_resolution=2000)

        result = detect_page(norm.image)
        print(
            f"{heic_path.name:<50} {result.confidence:>10.3f} {str(result.is_full_frame):>12}"
        )
