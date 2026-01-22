# Non-Bosch Animation Fix Summary

## Problem
The non-bosch sensor animation was appearing static/frozen in `acc_visualizer_v2.py`, with movement only visible during brief high-activity periods (like falls).

## Root Causes Identified

### 1. **Incorrect Import Path**
- The visualizer was importing calibration from `FD_ACCBARO` project instead of the local `sensor_calibration.py`
- The FD_ACCBARO version lacked the orientation correction matrix (ORIENTATION_MATRIX)
- This caused incorrect axis mapping for the 90° rotated non-bosch sensor

### 2. **Data Format Detection Issues**
- Some CSV files contain data already in g/mg units instead of raw LSB values
- The hardcoded scale factor (16384) didn't match all data formats
- After applying 0.25 scale factor to already-small values, the result was too tiny to visualize

### 3. **Missing Configuration System**
- No way to adjust sensor parameters (sampling rate, LSB/g ratio) for different hardware setups
- Bosch sensors can be configured at 25Hz or 50Hz, but this wasn't configurable

## Solutions Implemented

### 1. Fixed Import Priority
```python
# Now tries local directory FIRST (has orientation correction)
try:
    from sensor_calibration import transform_non_bosch_to_bosch
    CALIBRATION_AVAILABLE = True
    print("Using local sensor_calibration with orientation correction")
except ImportError:
    # Fallback to FD_ACCBARO only if local not found
    ...
```

### 2. Created Sensor Configuration System
**New file: `sensor_config.py`**
- Defines sensor-specific parameters (LSB/g, range, sampling rate)
- Provides `detect_data_format()` function to identify data format
- Easy to modify for different hardware configurations

**Example configuration:**
```python
BOSCH_CONFIG = {
    'name': 'BHI260AP',
    'lsb_per_g': 4096.0,    # LSB/g for ±8g range
    'range_g': 8,
    'sample_rate_hz': 50,   # Can be changed to 25, 50, 100, etc.
}

NON_BOSCH_CONFIG = {
    'name': 'LIS3DSH',
    'lsb_per_g': 16384.0,   # LSB/g for ±6g range
    'range_g': 6,
    'sample_rate_hz': 100,
}
```

### 3. Automatic Data Format Detection
The visualizer now automatically detects whether data is in:
- **Raw LSB units** (magnitude ~16384 for 1g) → Apply full calibration with scaling
- **g units** (magnitude ~1.0) → Apply only orientation correction, no scaling
- **mg units** (magnitude ~1000) → Apply only orientation correction, no scaling

**Detection logic:**
```python
sample_mag = sqrt(x² + y² + z²)

if sample_mag < 10:
    → g units (scale = 1.0)
elif sample_mag < 1000:
    → mg units (scale = 1000.0)
else:
    → raw LSB (scale = estimated from magnitude)
```

### 4. Improved Calibration Logic
```python
if sensor_scale > 100:  # Raw LSB data
    # Apply full transformation (orientation + scaling)
    bosch_values = (ORIENTATION_MATRIX @ non_bosch_values) * 0.25
    sensor_scale = 4096.0
else:  # g/mg units
    # Apply only 90° orientation correction
    bosch_x = -non_bosch_y
    bosch_y = non_bosch_x
    bosch_z = non_bosch_z
    # Keep original scale
```

## How to Use

### For Standard Setup (Default)
Just run the visualizer - it will auto-detect data format:
```bash
python acc_visualizer_v2.py your_data.csv
```

### For Different Sampling Rates
Edit `sensor_config.py`:
```python
# Change from 25Hz to 50Hz
BOSCH_CONFIG = {
    'name': 'BHI260AP',
    'lsb_per_g': 4096.0,
    'range_g': 8,
    'sample_rate_hz': 50,  # ← Change this
}
```

### For Different Sensor Ranges
Edit `sensor_config.py`:
```python
# Example: Non-Bosch with ±12g range
NON_BOSCH_CONFIG = {
    'name': 'LIS3DSH',
    'lsb_per_g': 8192.0,   # ← Adjust for ±12g range
    'range_g': 12,         # ← Change range
    'sample_rate_hz': 100,
}
```

## Debug Output

When loading non-bosch data, you'll now see:
```
  Non-Bosch data format: raw_lsb (magnitude: 16234.56)
  Non-Bosch samples: 50000
  Non-Bosch scale: 16234.56 LSB/g
  Applying calibration transform...
  Before calibration - Sample values: X=1234.0, Y=5678.0, Z=-890.0
  Before calibration - Scale: 16234.56
  After calibration - Sample values: X=-1419.5, Y=308.5, Z=-222.5
  After calibration - Scale: 4096.0
```

Or for g-unit data:
```
  Non-Bosch data format: g_units (magnitude: 1.02)
  Non-Bosch samples: 50000
  Non-Bosch scale: 1.0 LSB/g
  Applying calibration transform...
  WARNING: Data appears to be in g/mg units (scale=1.0), not raw LSB!
  Skipping transformation - applying only orientation correction...
```

## Files Modified

1. **acc_visualizer_v2.py**
   - Fixed import to prioritize local sensor_calibration.py
   - Added sensor_config import and usage
   - Improved calibration logic with format detection
   - Added comprehensive debug output

2. **sensor_config.py** (NEW)
   - Sensor-specific configurations
   - Data format detection function
   - Easy to modify for different hardware

3. **README.md**
   - Added "Sensor Configuration" section
   - Instructions for modifying sensor parameters

4. **NON_BOSCH_ANIMATION_FIX.md** (THIS FILE)
   - Documentation of the fix

## Testing Recommendations

1. **Test with your CSV file:**
   ```bash
   python acc_visualizer_v2.py "path/to/AIDAPT_100hz_nonBosch_Daria_FALL.csv"
   ```

2. **Check debug output** to verify:
   - Correct data format detected
   - Proper scale factor applied
   - Non-zero variance in transformed data

3. **Visual verification:**
   - Switch between Bosch and non-bosch sensors
   - Both should show similar orientation for same physical position
   - Animation should be smooth and responsive

4. **Compare movement patterns:**
   - Falls should produce similar visualization in both sensors
   - Stationary periods should show minimal movement (this is normal!)

## Why "Mostly Static" is Expected

For fall detection datasets:
- **90-95% of data:** Person is stationary or walking normally → minimal acceleration changes
- **5-10% of data:** Fall events, transitions, high activity → significant movement

This is **normal behavior**! The non-bosch sensor at 100Hz captures more samples, so the "active" portions appear even smaller as a percentage of total time.

If you want to focus on the interesting parts:
1. Use the time slider to jump to fall events
2. Adjust playback speed to slow down during falls
3. Check the magnitude plot to identify high-activity regions

## Sampling Rate Notes

- **100Hz non-bosch vs 25Hz bosch:** This difference doesn't affect animation
- The visualizer plays frames sequentially regardless of sampling rate
- Higher sampling rate = smoother animation = more frames to process
- If animation is too slow, increase the Speed slider

Date: 2026-01-22
