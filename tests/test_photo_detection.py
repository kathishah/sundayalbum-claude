"""Tests for photo detection and splitting."""

import numpy as np
import pytest

from src.photo_detection.detector import detect_photos, _order_corners
from src.photo_detection.splitter import split_photos
from src.photo_detection.classifier import classify_region


class TestPhotoDetection:
    """Tests for photo detection."""

    def test_detect_photos_single_image(self):
        """Test detecting a single photo (full frame)."""
        # Create a simple test image (white square on gray background)
        image = np.full((1000, 1000, 3), 0.5, dtype=np.float32)
        image[200:800, 200:800] = 1.0  # White square

        detections = detect_photos(image, min_area_ratio=0.1, max_count=5)

        # Should detect the white square
        assert len(detections) >= 1, "Should detect at least one photo"
        assert detections[0].bbox is not None
        assert detections[0].confidence > 0.0

    def test_detect_photos_multiple_images(self):
        """Test detecting multiple photos."""
        # Create test image with 2 white squares on gray background
        image = np.full((1000, 2000, 3), 0.3, dtype=np.float32)
        image[200:800, 200:800] = 1.0  # First square
        image[200:800, 1200:1800] = 1.0  # Second square

        detections = detect_photos(image, min_area_ratio=0.05, max_count=5)

        # Should detect 2 photos
        assert len(detections) >= 2, f"Should detect 2 photos, got {len(detections)}"

    def test_detect_photos_min_area_filter(self):
        """Test that small regions are filtered by min_area_ratio."""
        # Create test image with one large and one tiny square
        image = np.full((1000, 1000, 3), 0.3, dtype=np.float32)
        image[100:900, 100:900] = 1.0  # Large square (64% of image)
        image[50:70, 50:70] = 1.0  # Tiny square (0.04% of image)

        detections = detect_photos(image, min_area_ratio=0.05, max_count=5)

        # Should detect only the large square (tiny one filtered out)
        assert len(detections) == 1, f"Should detect 1 photo (filter tiny), got {len(detections)}"

    def test_detect_photos_decoration_filter(self):
        """Test that small decorations are filtered when real photos are present."""
        # Create test image with 2 large squares and 1 small decoration
        image = np.full((1000, 2000, 3), 0.3, dtype=np.float32)
        image[200:800, 200:800] = 1.0  # First photo (large)
        image[200:800, 1200:1800] = 1.0  # Second photo (large)
        image[50:150, 50:150] = 1.0  # Small decoration (< 30% of large photo)

        detections = detect_photos(image, min_area_ratio=0.01, max_count=5)

        # Should detect 2 photos (decoration filtered by 30% rule)
        assert len(detections) == 2, f"Should detect 2 photos, decoration filtered, got {len(detections)}"


class TestPhotoSplitting:
    """Tests for photo splitting."""

    def test_split_photos_single(self):
        """Test extracting a single photo."""
        # Create test image with white square
        image = np.full((1000, 1000, 3), 0.5, dtype=np.float32)
        image[200:800, 200:800] = 1.0

        # Create mock detection
        from src.photo_detection.detector import PhotoDetection
        detection = PhotoDetection(
            bbox=(200, 200, 800, 800),
            corners=np.array([[200, 200], [800, 200], [800, 800], [200, 800]], dtype=np.float32),
            confidence=0.95,
            orientation="square",
            area_ratio=0.36,
            contour=np.array([])
        )

        photos = split_photos(image, [detection])

        assert len(photos) == 1
        assert photos[0].shape[0] > 0 and photos[0].shape[1] > 0

    def test_split_photos_multiple(self):
        """Test extracting multiple photos."""
        # Create test image with 2 white squares
        image = np.full((1000, 2000, 3), 0.3, dtype=np.float32)
        image[200:800, 200:800] = 1.0
        image[200:800, 1200:1800] = 1.0

        # Create mock detections
        from src.photo_detection.detector import PhotoDetection
        detections = [
            PhotoDetection(
                bbox=(200, 200, 800, 800),
                corners=np.array([[200, 200], [800, 200], [800, 800], [200, 800]], dtype=np.float32),
                confidence=0.95,
                orientation="square",
                area_ratio=0.18,
                contour=np.array([])
            ),
            PhotoDetection(
                bbox=(1200, 200, 1800, 800),
                corners=np.array([[1200, 200], [1800, 200], [1800, 800], [1200, 800]], dtype=np.float32),
                confidence=0.95,
                orientation="square",
                area_ratio=0.18,
                contour=np.array([])
            ),
        ]

        photos = split_photos(image, detections)

        assert len(photos) == 2
        for photo in photos:
            assert photo.shape[0] > 0 and photos[0].shape[1] > 0


class TestCornerOrdering:
    """Tests for corner ordering."""

    def test_order_corners_clockwise(self):
        """Test that corners are ordered clockwise from top-left."""
        # Corners in random order
        corners = np.array([
            [100, 100],  # top-left
            [200, 200],  # bottom-right
            [200, 100],  # top-right
            [100, 200],  # bottom-left
        ], dtype=np.float32)

        ordered = _order_corners(corners)

        # Expected order: TL, TR, BR, BL
        assert ordered.shape == (4, 2)
        np.testing.assert_array_equal(ordered[0], [100, 100])  # top-left
        np.testing.assert_array_equal(ordered[1], [200, 100])  # top-right
        np.testing.assert_array_equal(ordered[2], [200, 200])  # bottom-right
        np.testing.assert_array_equal(ordered[3], [100, 200])  # bottom-left


class TestRegionClassifier:
    """Tests for region classification."""

    def test_classify_photo(self):
        """Test classifying a normal photo region."""
        # Photo-like region with varied colors
        region = np.random.rand(500, 500, 3).astype(np.float32)
        region_type = classify_region(region, bbox_area=250000, page_area=1000000)

        assert region_type == "photo"

    def test_classify_decoration_small(self):
        """Test classifying a small decoration."""
        # Small uniform region
        region = np.full((20, 20, 3), 0.5, dtype=np.float32)
        region_type = classify_region(region, bbox_area=400, page_area=1000000)

        assert region_type == "decoration"

    def test_classify_caption_elongated(self):
        """Test classifying an elongated caption region."""
        # Very elongated region (text-like)
        region = np.random.rand(20, 200, 3).astype(np.float32)
        region_type = classify_region(region, bbox_area=4000, page_area=1000000)

        assert region_type == "caption"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
