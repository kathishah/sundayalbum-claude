"""Tests for image loading functionality."""

import pytest
import numpy as np
from pathlib import Path

from src.preprocessing.loader import load_image, ImageMetadata


# Test data directory
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test-images"


@pytest.fixture
def test_images_exist():
    """Check if test images directory exists and has files."""
    if not TEST_IMAGES_DIR.exists():
        pytest.skip(f"Test images directory not found: {TEST_IMAGES_DIR}")

    # Check for at least one HEIC and one DNG file
    heic_files = list(TEST_IMAGES_DIR.glob("*.HEIC")) + list(TEST_IMAGES_DIR.glob("*.heic"))
    dng_files = list(TEST_IMAGES_DIR.glob("*.DNG")) + list(TEST_IMAGES_DIR.glob("*.dng"))

    if not heic_files:
        pytest.skip("No HEIC files found in test-images directory")
    if not dng_files:
        pytest.skip("No DNG files found in test-images directory")

    return heic_files, dng_files


def test_load_heic(test_images_exist):
    """Test loading HEIC image."""
    heic_files, _ = test_images_exist

    # Use the first HEIC file
    heic_path = heic_files[0]

    # Load image
    image, metadata = load_image(str(heic_path))

    # Check array properties
    assert isinstance(image, np.ndarray), "Image should be numpy array"
    assert image.dtype == np.float32, "Image should be float32"
    assert image.ndim == 3, "Image should be 3D array (H, W, C)"
    assert image.shape[2] == 3, "Image should have 3 channels (RGB)"

    # Check value range
    assert image.min() >= 0.0, "Image values should be >= 0.0"
    assert image.max() <= 1.0, "Image values should be <= 1.0"

    # Check metadata
    assert isinstance(metadata, ImageMetadata), "Metadata should be ImageMetadata object"
    assert metadata.format == "HEIC", "Format should be HEIC"
    assert metadata.bit_depth == 8, "HEIC should have 8-bit depth"
    assert len(metadata.original_size) == 2, "Original size should be (width, height)"
    assert metadata.original_size[0] > 0, "Width should be positive"
    assert metadata.original_size[1] > 0, "Height should be positive"

    # Check that image dimensions are reasonable (at least 10MP)
    total_pixels = image.shape[0] * image.shape[1]
    assert total_pixels > 10_000_000, f"HEIC should be high resolution, got {total_pixels} pixels"

    print(f"✓ HEIC loaded: {heic_path.name} - {image.shape[1]}x{image.shape[0]} ({total_pixels/1e6:.1f}MP)")


def test_load_dng(test_images_exist):
    """Test loading DNG image."""
    _, dng_files = test_images_exist

    # Use the first DNG file
    dng_path = dng_files[0]

    # Load image
    image, metadata = load_image(str(dng_path))

    # Check array properties
    assert isinstance(image, np.ndarray), "Image should be numpy array"
    assert image.dtype == np.float32, "Image should be float32"
    assert image.ndim == 3, "Image should be 3D array (H, W, C)"
    assert image.shape[2] == 3, "Image should have 3 channels (RGB)"

    # Check value range
    assert image.min() >= 0.0, "Image values should be >= 0.0"
    assert image.max() <= 1.0, "Image values should be <= 1.0"

    # Check metadata
    assert isinstance(metadata, ImageMetadata), "Metadata should be ImageMetadata object"
    assert metadata.format == "DNG", "Format should be DNG"
    assert metadata.bit_depth == 16, "DNG should have 16-bit depth"
    assert len(metadata.original_size) == 2, "Original size should be (width, height)"

    # Check that image dimensions are reasonable (at least 10MP)
    total_pixels = image.shape[0] * image.shape[1]
    assert total_pixels > 10_000_000, f"DNG should be high resolution, got {total_pixels} pixels"

    print(f"✓ DNG loaded: {dng_path.name} - {image.shape[1]}x{image.shape[0]} ({total_pixels/1e6:.1f}MP)")


def test_heic_vs_dng_same_scene(test_images_exist):
    """Test that HEIC and DNG of the same scene produce similar results."""
    heic_files, dng_files = test_images_exist

    # Find a matching pair (e.g., IMG_cave_normal.HEIC and IMG_cave_prores.DNG)
    for heic_path in heic_files:
        # Extract scene name (e.g., "cave" from "IMG_cave_normal.HEIC")
        scene_name = heic_path.stem.replace("IMG_", "").replace("_normal", "")

        # Look for matching DNG
        matching_dng = None
        for dng_path in dng_files:
            if scene_name in dng_path.stem:
                matching_dng = dng_path
                break

        if matching_dng:
            # Load both
            heic_image, heic_meta = load_image(str(heic_path))
            dng_image, dng_meta = load_image(str(matching_dng))

            # DNG should have same or higher resolution
            heic_pixels = heic_image.shape[0] * heic_image.shape[1]
            dng_pixels = dng_image.shape[0] * dng_image.shape[1]

            assert dng_pixels >= heic_pixels, "DNG should have same or more pixels than HEIC"

            # Aspect ratio should be similar
            heic_aspect = heic_image.shape[1] / heic_image.shape[0]
            dng_aspect = dng_image.shape[1] / dng_image.shape[0]

            aspect_diff = abs(heic_aspect - dng_aspect)
            assert aspect_diff < 0.01, f"Aspect ratios should match (diff: {aspect_diff})"

            print(f"✓ Matched pair: {heic_path.name} ({heic_pixels/1e6:.1f}MP) vs {matching_dng.name} ({dng_pixels/1e6:.1f}MP)")

            return  # Test passed with first matching pair

    pytest.skip("No matching HEIC/DNG pairs found")


def test_all_heic_files_load(test_images_exist):
    """Test that all HEIC files load without error."""
    heic_files, _ = test_images_exist

    loaded_count = 0
    for heic_path in heic_files:
        try:
            image, metadata = load_image(str(heic_path))
            assert image is not None
            assert metadata is not None
            loaded_count += 1
            print(f"✓ {heic_path.name}: {image.shape[1]}x{image.shape[0]}")
        except Exception as e:
            pytest.fail(f"Failed to load {heic_path}: {e}")

    assert loaded_count == len(heic_files), f"Should load all {len(heic_files)} HEIC files"


def test_all_dng_files_load(test_images_exist):
    """Test that all DNG files load without error."""
    _, dng_files = test_images_exist

    loaded_count = 0
    for dng_path in dng_files:
        try:
            image, metadata = load_image(str(dng_path))
            assert image is not None
            assert metadata is not None
            loaded_count += 1
            print(f"✓ {dng_path.name}: {image.shape[1]}x{image.shape[0]}")
        except Exception as e:
            pytest.fail(f"Failed to load {dng_path}: {e}")

    assert loaded_count == len(dng_files), f"Should load all {len(dng_files)} DNG files"


def test_file_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_image("nonexistent_file.heic")


def test_unsupported_format():
    """Test that ValueError is raised for unsupported format."""
    # Create a temporary file with unsupported extension
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported image format"):
            load_image(temp_path)
    finally:
        Path(temp_path).unlink()
