# Sensor Orientation Fix Summary

## Problem Identified

The two accelerometer sensors are mounted with **different physical orientations** on the wrist, rotated 90° relative to each other in the X-Y plane. This was not being properly accounted for in the calibration and visualization code.

## Sensor Orientations (from Hardware)

### BOSCH (BHI260AP) - Baseline Sensor
```
X-axis: negative = finger-tip,  positive = shoulder
Y-axis: negative = thumb,       positive = pinky
Z-axis: negative = air/nail,    positive = thigh/palm
```

### NON-BOSCH (LIS3DSH) - Rotated 90° from Bosch
```
X-axis: negative = thumb,       positive = pinky
Y-axis: negative = shoulder,    positive = finger-tip
Z-axis: negative = air/nail,    positive = thigh/palm
```

**Key Difference**: The sensors are rotated 90° in the X-Y plane!

## Changes Made

### 1. sensor_calibration.py

**Added ORIENTATION_MATRIX** to handle the 90° rotation:
```python
ORIENTATION_MATRIX = np.array([
    [ 0, -1,  0],  # Bosch_X = -Non-bosch_Y
    [ 1,  0,  0],  # Bosch_Y = +Non-bosch_X
    [ 0,  0,  1],  # Bosch_Z = +Non-bosch_Z
])
```

**Transformation Steps**:
1. **Orientation Correction**: Apply ORIENTATION_MATRIX to map axes correctly
2. **Scale Correction**: Convert from 16384 LSB/g (non-bosch) to 4096 LSB/g (bosch)
3. **Result**: Non-bosch data that matches what Bosch would have measured

**Changes**:
- Added `ORIENTATION_MATRIX` for proper axis mapping
- Added `SCALE_FACTOR` = 4096/16384 = 0.25
- Renamed old matrix to `TRANSFORM_MATRIX_LEGACY`
- Updated `transform_non_bosch_to_bosch()` to use orientation correction by default
- Updated `SensorCalibrator` class to support both new and legacy modes
- Added detailed documentation explaining the hardware differences

### 2. acc_visualizer_v2.py

**Updated Documentation**:
- Fixed header comments to reflect correct Bosch orientation
- Updated `acc_to_rotation()` function comments
- Corrected axis mapping: `gx = sy` (was `gx = -sy`)
- Updated console output to show correct axes

**Axis Mapping (Bosch → Model)**:
```python
gx = sy   # Model X (pinky) from +Sensor Y
gy = -sx  # Model Y (fingers) from -Sensor X
gz = -sz  # Model Z (back) from -Sensor Z
```

## What This Fixes

1. **Visualization Accuracy**: The hand model now correctly responds to sensor movements for BOTH sensors
2. **Calibration Accuracy**: Non-bosch data is properly transformed to match bosch reference frame
3. **Code Documentation**: All comments and docstrings now reflect actual hardware orientation

## Backward Compatibility

- Legacy transformation still available via `use_orientation_correction=False` parameter
- Old code importing `TRANSFORM_MATRIX` will get `ORIENTATION_MATRIX` (correct behavior)
- Both transformation modes supported in `SensorCalibrator` class

## Testing Recommendations

1. **Visual Test**: Run visualizer with both sensors - hand should orient identically for same physical position
2. **Static Test**: Place device flat → should show "Nails up" for both sensors
3. **Rotation Test**: Rotate wrist palm up → should show "Palm up" for both sensors
4. **Comparison Test**: Same movement recorded by both sensors should produce similar visualization

## Files Modified

1. `sensor_calibration.py` - Added orientation correction
2. `acc_visualizer_v2.py` - Fixed axis mapping for Bosch sensor
3. `ORIENTATION_FIX_SUMMARY.md` - This file (documentation)
