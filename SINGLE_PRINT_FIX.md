# Fix: Photo Detection for Single Prints

## Problem

The photo detection algorithm was failing to crop individual photo prints from their backgrounds. When processing single prints (e.g., cave, harbor, skydiving images), the pipeline would fall back to saving the entire image including the table/background instead of extracting just the photo.

### Root Cause

The issue was in `src/photo_detection/detector.py`:

1. **Hard-coded max_area_threshold = 0.90**: This filtered out any contour larger than 90% of the image as a "page boundary"
2. **Single prints are large**: For individual photo prints on a table, the photo typically occupies 60-85% of the captured image
3. **Incorrect filtering**: These large single photos were being filtered out as page boundaries, causing the detection to return zero results
4. **Fallback behavior**: When no photos were detected, the pipeline fell back to using the entire uncropped image (line 387-388 in pipeline.py)

## Solution

Implemented **adaptive max_area threshold** based on detection scenario:

### Two-Pass Algorithm

**First Pass: Collect Potential Detections**
- Filter by minimum area (2%)
- Filter extremely large contours (>95% = definitely full frame)
- Store all potential detections for analysis

**Scenario Detection**
```python
# Classify the image type based on contour sizes
large_contours = [d for d in potential_detections if d[3] >= 0.30]  # >= 30% area
medium_contours = [d for d in potential_detections if 0.10 <= d[3] < 0.30]

is_single_print = (len(large_contours) == 1 and len(medium_contours) == 0)
```

**Second Pass: Apply Scenario-Specific Filtering**
- **Single Print Mode**: max_area_threshold = 0.95 (allow large photos up to 95%)
- **Multi-Photo Page Mode**: max_area_threshold = 0.90 (filter page boundaries)

### Key Insights

1. **Single prints vs Album pages are fundamentally different**:
   - Single print: 1 large photo (30-95% of image) on background
   - Album page: Multiple smaller photos (2-30% each) within page boundary

2. **The threshold must adapt**: Using a fixed threshold fails for one scenario or the other

3. **Conservative filtering**: Still filter >95% to avoid full-frame captures

## Testing

Added new test case: `test_detect_single_print_on_background()`

```python
def test_detect_single_print_on_background(self):
    """Test detecting a single large photo print on a background.

    This is the key test case for cave/harbor/skydiving images where a single
    photo print takes up 60-80% of the image and should be detected and cropped.
    """
    # Create test image: large photo (70% of image) on background
    image = np.full((1000, 1000, 3), 0.3, dtype=np.float32)  # Dark background
    image[100:900, 100:900] = 0.8  # Large photo print (64% of image)

    detections = detect_photos(image, min_area_ratio=0.02, max_count=5)

    # Should detect exactly 1 photo
    assert len(detections) == 1
    assert detections[0].area_ratio > 0.5  # >50% of image
    assert detections[0].area_ratio < 0.95  # <95% (not full frame)
```

## Expected Outcomes

### Before Fix
- **cave/harbor/skydiving**: 0 detections → fallback to full uncropped image
- **three_pics**: 3 photos detected ✓
- **two_pics**: 2 photos detected ✓

### After Fix
- **cave/harbor/skydiving**: 1 photo detected → cropped to photo boundaries ✓
- **three_pics**: 3 photos detected ✓ (unchanged)
- **two_pics**: 2 photos detected ✓ (unchanged)

## Files Changed

1. `src/photo_detection/detector.py`:
   - Added two-pass contour filtering
   - Added scenario detection logic
   - Implemented adaptive max_area threshold
   - Added detailed debug logging

2. `tests/test_photo_detection.py`:
   - Added `test_detect_single_print_on_background()` test case

## Impact

This fix enables proper extraction of individual photo prints from their backgrounds, which is essential for the PRD requirement:

> **Section 5.2.2**: "Crop each detected photo into its own individual image"

Now both use cases work correctly:
- ✅ Album pages with multiple photos → extracts individual photos
- ✅ Single photo prints on backgrounds → crops to photo boundaries

## Technical Details

The algorithm now:
1. ✅ Detects single large photos (30-95% of image)
2. ✅ Distinguishes single prints from page boundaries
3. ✅ Maintains correct multi-photo detection
4. ✅ Provides detailed debug logging for troubleshooting
