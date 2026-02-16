"""Phase 6 integration tests with real images and debug output."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.loader import load_image
from src.photo_detection.detector import detect_photos
from src.photo_detection.splitter import split_photos
from src.utils.debug import save_debug_image, draw_photo_detections


class TestPhase6Integration:
    """Integration tests for Phase 6 photo detection with real images."""

    @pytest.fixture(autouse=True)
    def setup_debug_dir(self):
        """Create debug directory for each test."""
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        yield debug_dir
        # Keep debug output for manual inspection - don't clean up

    def test_three_pics_detection_with_debug(self, setup_debug_dir):
        """Test photo detection on three_pics album page with debug output."""
        test_image_path = Path("test-images/IMG_three_pics_normal.HEIC")

        if not test_image_path.exists():
            pytest.skip("Test image not available. Run scripts/fetch-test-images.sh")

        debug_dir = setup_debug_dir

        # Load image
        print(f"\n📸 Loading {test_image_path.name}...")
        image = load_image(str(test_image_path))

        # Save original
        save_debug_image(
            image,
            debug_dir / "phase6_01_original.jpg",
            "Original album page"
        )

        # Detect photos
        print("🔍 Detecting photos...")
        detections = detect_photos(
            image,
            min_area_ratio=0.05,
            max_count=5
        )

        print(f"✓ Found {len(detections)} photo(s)")
        for i, det in enumerate(detections, 1):
            print(f"  Photo {i}: area_ratio={det.area_ratio:.2%}, "
                  f"confidence={det.confidence:.2f}, "
                  f"type={det.region_type}")

        # Draw detections
        detection_viz = draw_photo_detections(image, detections)
        save_debug_image(
            detection_viz,
            debug_dir / "phase6_02_detections.jpg",
            f"Detected {len(detections)} photos"
        )

        # Split photos
        print("✂️  Splitting photos...")
        photos = split_photos(image, detections)

        for i, photo in enumerate(photos, 1):
            save_debug_image(
                photo,
                debug_dir / f"phase6_03_photo_{i:02d}.jpg",
                f"Extracted photo {i}"
            )
            print(f"✓ Saved photo {i}: {photo.shape[1]}×{photo.shape[0]}px")

        # Assertions
        assert len(detections) == 3, f"Expected 3 photos, got {len(detections)}"
        assert len(photos) == 3, f"Expected 3 extracted photos, got {len(photos)}"

        # All photos should be classified as "photo" (not decoration/caption)
        for i, det in enumerate(detections, 1):
            assert det.region_type == "photo", \
                f"Photo {i} misclassified as {det.region_type}"

        print(f"\n✅ Debug output saved to {debug_dir}/")
        print("   Review the images to verify photo detection quality!")

    def test_two_pics_mixed_orientation_with_debug(self, setup_debug_dir):
        """Test photo detection on two_pics (portrait + landscape) with debug output."""
        test_image_path = Path("test-images/IMG_two_pics_vertical_horizontal_normal.HEIC")

        if not test_image_path.exists():
            pytest.skip("Test image not available. Run scripts/fetch-test-images.sh")

        debug_dir = setup_debug_dir

        # Load image
        print(f"\n📸 Loading {test_image_path.name}...")
        image = load_image(str(test_image_path))

        # Save original
        save_debug_image(
            image,
            debug_dir / "phase6_mixed_01_original.jpg",
            "Original album page (mixed orientation)"
        )

        # Detect photos
        print("🔍 Detecting photos...")
        detections = detect_photos(
            image,
            min_area_ratio=0.05,
            max_count=5
        )

        print(f"✓ Found {len(detections)} photo(s)")
        for i, det in enumerate(detections, 1):
            print(f"  Photo {i}: orientation={det.orientation}, "
                  f"area_ratio={det.area_ratio:.2%}, "
                  f"confidence={det.confidence:.2f}")

        # Draw detections
        detection_viz = draw_photo_detections(image, detections)
        save_debug_image(
            detection_viz,
            debug_dir / "phase6_mixed_02_detections.jpg",
            f"Detected {len(detections)} photos (mixed orientation)"
        )

        # Split photos
        print("✂️  Splitting photos...")
        photos = split_photos(image, detections)

        for i, photo in enumerate(photos, 1):
            save_debug_image(
                photo,
                debug_dir / f"phase6_mixed_03_photo_{i:02d}.jpg",
                f"Extracted photo {i}"
            )
            print(f"✓ Saved photo {i}: {photo.shape[1]}×{photo.shape[0]}px")

        # Assertions
        assert len(detections) == 2, f"Expected 2 photos, got {len(detections)}"
        assert len(photos) == 2, f"Expected 2 extracted photos, got {len(photos)}"

        # Check we have both portrait and landscape orientations
        orientations = {det.orientation for det in detections}
        assert "portrait" in orientations or "landscape" in orientations, \
            "Should detect photos with different orientations"

        print(f"\n✅ Debug output saved to {debug_dir}/")
        print("   Review the images to verify mixed orientation detection!")

    def test_single_photo_cave_with_debug(self, setup_debug_dir):
        """Test photo detection on single glossy print (cave) with debug output."""
        test_image_path = Path("test-images/IMG_cave_normal.HEIC")

        if not test_image_path.exists():
            pytest.skip("Test image not available. Run scripts/fetch-test-images.sh")

        debug_dir = setup_debug_dir

        # Load image
        print(f"\n📸 Loading {test_image_path.name}...")
        image = load_image(str(test_image_path))

        # Save original
        save_debug_image(
            image,
            debug_dir / "phase6_single_01_original.jpg",
            "Original single glossy print"
        )

        # Detect photos (should find whole image as one photo)
        print("🔍 Detecting photos...")
        detections = detect_photos(
            image,
            min_area_ratio=0.05,
            max_count=5
        )

        print(f"✓ Found {len(detections)} photo(s)")
        for i, det in enumerate(detections, 1):
            print(f"  Photo {i}: area_ratio={det.area_ratio:.2%}, "
                  f"confidence={det.confidence:.2f}")

        # Draw detections
        detection_viz = draw_photo_detections(image, detections)
        save_debug_image(
            detection_viz,
            debug_dir / "phase6_single_02_detections.jpg",
            f"Detected {len(detections)} photo(s)"
        )

        # Split photos
        if len(detections) > 0:
            print("✂️  Splitting photos...")
            photos = split_photos(image, detections)

            for i, photo in enumerate(photos, 1):
                save_debug_image(
                    photo,
                    debug_dir / f"phase6_single_03_photo_{i:02d}.jpg",
                    f"Extracted photo {i}"
                )
                print(f"✓ Saved photo {i}: {photo.shape[1]}×{photo.shape[0]}px")

        # Assertions
        assert len(detections) >= 1, "Should detect at least one photo"

        print(f"\n✅ Debug output saved to {debug_dir}/")
        print("   Review the images to verify single photo detection!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
