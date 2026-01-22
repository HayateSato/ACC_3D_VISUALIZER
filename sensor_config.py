"""
Sensor Configuration Module

This file defines sensor-specific parameters for different accelerometers.
Modify these values based on your specific hardware configuration.
"""

# =============================================================================
# SENSOR SPECIFICATIONS
# =============================================================================

# Bosch BHI260AP Configuration
BOSCH_CONFIG = {
    'name': 'BHI260AP',
    'lsb_per_g': 4096.0,     # LSB/g for ±8g range
    'range_g': 8,            # ±8g
    'sample_rate_hz': 25,    # Default sampling rate
    'description': 'Bosch BHI260AP IMU with ±8g range'
}

# Non-Bosch LIS3DSH Configuration
NON_BOSCH_CONFIG = {
    'name': 'LIS3DSH',
    'lsb_per_g': 16384.0,    # LSB/g for ±6g range
    'range_g': 6,            # ±6g
    'sample_rate_hz': 100,   # Default sampling rate
    'description': 'LIS3DSH accelerometer with ±6g range'
}

# Alternative configurations for different setups
# Uncomment and modify if your hardware uses different settings

# # Bosch 50Hz configuration
# BOSCH_CONFIG = {
#     'name': 'BHI260AP',
#     'lsb_per_g': 4096.0,
#     'range_g': 8,
#     'sample_rate_hz': 50,    # 50Hz sampling
#     'description': 'Bosch BHI260AP IMU at 50Hz'
# }

# # Non-Bosch with different range
# NON_BOSCH_CONFIG = {
#     'name': 'LIS3DSH',
#     'lsb_per_g': 8192.0,     # LSB/g for ±12g range
#     'range_g': 12,
#     'sample_rate_hz': 100,
#     'description': 'LIS3DSH accelerometer with ±12g range'
# }

# =============================================================================
# DATA FORMAT DETECTION
# =============================================================================

def detect_data_format(acc_x, acc_y, acc_z, expected_lsb_per_g=16384.0):
    """
    Detect if accelerometer data is in raw LSB units or already in g units.

    Args:
        acc_x, acc_y, acc_z: Sample accelerometer values (can be single values or arrays)
        expected_lsb_per_g: Expected LSB/g ratio for raw data

    Returns:
        dict with:
            - 'format': 'raw_lsb', 'g_units', or 'mg_units'
            - 'estimated_scale': Estimated scale factor
            - 'magnitude': Sample magnitude
    """
    import numpy as np

    # Calculate magnitude of first sample
    if isinstance(acc_x, (list, np.ndarray)):
        sample_x, sample_y, sample_z = acc_x[0], acc_y[0], acc_z[0]
    else:
        sample_x, sample_y, sample_z = acc_x, acc_y, acc_z

    magnitude = np.sqrt(sample_x**2 + sample_y**2 + sample_z**2)

    # Determine format based on magnitude
    if magnitude < 10:
        # Likely in g units (gravity ≈ 1.0)
        return {
            'format': 'g_units',
            'estimated_scale': 1.0,
            'magnitude': magnitude,
            'confidence': 'high' if magnitude > 0.5 and magnitude < 2.0 else 'medium'
        }
    elif magnitude < 1000:
        # Likely in milli-g units (gravity ≈ 1000 mg)
        return {
            'format': 'mg_units',
            'estimated_scale': 1000.0,
            'magnitude': magnitude,
            'confidence': 'high' if magnitude > 500 and magnitude < 2000 else 'medium'
        }
    else:
        # Likely in raw LSB units
        estimated_lsb = magnitude  # Rough estimate assuming at rest (1g)
        return {
            'format': 'raw_lsb',
            'estimated_scale': estimated_lsb,
            'magnitude': magnitude,
            'confidence': 'high' if abs(estimated_lsb - expected_lsb_per_g) < expected_lsb_per_g * 0.5 else 'low'
        }


def get_sensor_config(sensor_type='bosch'):
    """
    Get configuration for specified sensor type.

    Args:
        sensor_type: 'bosch' or 'non_bosch'

    Returns:
        Configuration dictionary
    """
    if sensor_type.lower() == 'bosch':
        return BOSCH_CONFIG.copy()
    else:
        return NON_BOSCH_CONFIG.copy()


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("Sensor Configuration")
    print("=" * 60)
    print("\nBosch Sensor:")
    for key, value in BOSCH_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nNon-Bosch Sensor:")
    for key, value in NON_BOSCH_CONFIG.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Scale Factor for Calibration:")
    scale_factor = BOSCH_CONFIG['lsb_per_g'] / NON_BOSCH_CONFIG['lsb_per_g']
    print(f"  {NON_BOSCH_CONFIG['lsb_per_g']} LSB/g → {BOSCH_CONFIG['lsb_per_g']} LSB/g")
    print(f"  Scale Factor: {scale_factor}")
