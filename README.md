# 3D Accelerometer Visualizer

Interactive 3D visualization tool for wrist-worn accelerometer data. Shows how a hand/wrist moves during activities like falls based on accelerometer readings.

## Features

- **3D Hand Model**: Animated hand that rotates based on accelerometer orientation
- **Dual Sensor Support**: Switch between Bosch and non-Bosch accelerometer sensors
- **Interactive Timeline**: Slider to navigate through time, Play/Pause animation
- **Motion Trail**: Visual trail showing recent wrist movement path
- **Acceleration Vector**: Shows direction and magnitude of current acceleration
- **Real-time Values**: Display of X, Y, Z values and G-force

## Installation

```bash
cd ACC_Visualizer
pip install -r requirements.txt
```

## Usage

### Basic Version
```bash
python acc_visualizer.py
# Then enter your CSV file path when prompted
```

### Enhanced Version (Recommended)
```bash
python acc_visualizer_enhanced.py
# Or provide path directly:
python acc_visualizer_enhanced.py "C:/path/to/your/data.csv"
```

## Input Data Format

The tool expects CSV files in SmarkoApp format with these columns:

| Column | Description |
|--------|-------------|
| `timestamp` | Epoch time in milliseconds |
| `is_accelerometer` | 1 if row has non-Bosch ACC data |
| `acc_x`, `acc_y`, `acc_z` | Non-Bosch accelerometer values |
| `is_accelerometer_bosch` | 1 if row has Bosch ACC data |
| `bosch_acc_x`, `bosch_acc_y`, `bosch_acc_z` | Bosch accelerometer values |

Example CSV path:
```
C:\Users\HayateSato\Documents\6G\FD_ACCBARO\results\fall_data_exports\20260120\SmarkoApp_csv\AIDAPT_25hz_Bosch_isa_FALL.csv
```

## Controls

| Control | Description |
|---------|-------------|
| **Time Slider** | Drag to navigate through the recording |
| **Play/Pause** | Start/stop automatic animation |
| **Speed Slider** | Adjust animation playback speed |
| **Sensor Radio** | Switch between Bosch and non-Bosch sensors |
| **Trail Checkbox** | Show/hide motion trail |
| **Vector Checkbox** | Show/hide acceleration vector |
| **Mouse Drag on 3D** | Rotate camera view |

## Sensor Configuration

### Configuring Different Hardware

If your sensors use different sampling rates or sensitivity ranges, edit `sensor_config.py`:

```python
# Example: Bosch sensor at 50Hz instead of 25Hz
BOSCH_CONFIG = {
    'name': 'BHI260AP',
    'lsb_per_g': 4096.0,     # LSB/g for ±8g range
    'range_g': 8,
    'sample_rate_hz': 50,    # Change this to match your setup
    'description': 'Bosch BHI260AP IMU at 50Hz'
}
```

The visualizer automatically detects data format (raw LSB vs g units) and adjusts accordingly.

## Understanding the Visualization

### Sensor Orientation (Bosch on LEFT wrist)
- **X-axis**: Finger ↔ Shoulder (negative = toward fingers, positive = toward shoulder)
- **Y-axis**: Pinky ↔ Thumb (positive = toward pinky, negative = toward thumb)
- **Z-axis**: Nails ↔ Palm (negative = toward nails/back of hand, positive = toward palm/thigh)

### Hand Orientation
The hand model rotates based on accelerometer readings. When stationary:
- Gravity (~4096 for Bosch, ~16384 for non-Bosch raw units = 1g) points downward
- The hand orientation shows which direction is "down"

### Motion Trail
The purple/yellow trail shows recent wrist positions, helping visualize:
- Direction of movement
- Speed (trail points closer = slower, farther = faster)
- Movement patterns during falls

### Acceleration Vector
The purple arrow shows:
- Direction of current acceleration
- Length proportional to magnitude
- During a fall, this vector changes rapidly

### G-Force Indicator
- Normal stationary: ~1g
- Light activity: 1-2g
- Impact/fall: >3g (shown in red)

## Files

| File | Description |
|------|-------------|
| `acc_visualizer.py` | Basic version with simple hand model |
| `acc_visualizer_enhanced.py` | Enhanced version with trail and better graphics |
| `requirements.txt` | Python dependencies |

## Tips for Presentation

1. **Start at a calm moment**: Begin the animation when the person is standing still
2. **Watch the trail**: Notice how it changes during a fall
3. **Compare sensors**: Use the radio buttons to see differences between Bosch and non-Bosch
4. **Pause at key moments**: Stop at impact to show the G-force spike
5. **Rotate the view**: Drag on the 3D plot to show different angles

## Troubleshooting

### "No accelerometer data found"
- Check that your CSV has the required columns
- Ensure `is_accelerometer` or `is_accelerometer_bosch` columns have value 1 for ACC rows

### Animation is slow
- Reduce the Speed slider
- Close other applications
- Use a smaller time window (stop and restart at different points)

### Hand doesn't rotate correctly
- This is normal for very high acceleration events
- The visualization is optimized for orientation, not absolute position
