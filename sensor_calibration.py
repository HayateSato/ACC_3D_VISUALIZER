"""
Sensor Calibration Module - Transform non_bosch accelerometer data to bosch-equivalent values.

This module provides transformation functions to convert accelerometer readings from
the non_bosch sensor to values that approximate what the bosch sensor would measure.
This allows models trained on bosch sensor data to be used with non_bosch sensor input.

IMPORTANT - Hardware Orientation Differences:
    The two sensors are mounted with DIFFERENT orientations on the wrist:

    BOSCH (BHI260AP):
        X: negative = finger-tip,  positive = shoulder
        Y: negative = thumb,       positive = pinky
        Z: negative = air/nail,    positive = thigh/palm

    NON-BOSCH (LIS3DSH):
        X: negative = thumb,       positive = pinky
        Y: negative = shoulder,    positive = finger-tip
        Z: negative = air/nail,    positive = thigh/palm

    The sensors are rotated 90° relative to each other in the X-Y plane!

Calibration Method:
    The transformation accounts for both the orientation difference and sensor
    characteristics using least squares regression on time-aligned sensor pairs:

        bosch_values = ORIENTATION_CORRECTION @ TRANSFORM_MATRIX @ non_bosch_values + TRANSFORM_OFFSET

    Where:
        - ORIENTATION_CORRECTION handles the 90° rotation between sensors
        - TRANSFORM_MATRIX handles scaling and minor adjustments
        - TRANSFORM_OFFSET handles bias differences

Calibration Accuracy:
    Based on 9,991 time-aligned sample pairs from 4 recording sessions:

    Axis    RMSE        R²
    X       1683.9      0.6701
    Y       1392.9      0.3888
    Z       1484.0      0.5222

    Magnitude correlation: 0.4605

    Note: The transformation provides a moderate approximation. The R² values indicate
    that 39-67% of variance in bosch readings can be explained by non_bosch readings.
    This is useful for experimental evaluation but results may not match bosch sensor
    performance exactly.

Usage:
    from app.sensor_calibration import transform_non_bosch_to_bosch, SensorCalibrator

    # Single sample
    bosch_x, bosch_y, bosch_z = transform_non_bosch_to_bosch(acc_x, acc_y, acc_z)

    # Batch transform
    calibrator = SensorCalibrator()
    acc_data_calibrated = calibrator.transform(acc_data)  # Shape: (3, N)

Data Source:
    Calibration computed from:
    - AIDAPT_25hz_Bosch_isa_FALL.csv
    - AIDAPT_25hz_Bosch_isa_FALL_2.csv
    - AIDAPT_25hz_Bosch_Daria_FALL.csv
    - AIDAPT_25hz_Bosch_armSwingUp.csv

Date: 2026-01-20
"""

import numpy as np
from typing import Tuple, Union


# =============================================================================
# CALIBRATION PARAMETERS
# =============================================================================

# ORIENTATION CORRECTION MATRIX
# Handles the 90° rotation difference between the two sensors
# Mapping from non-bosch axes to bosch axes:
#   Bosch_X (finger↔shoulder) = Non-bosch_Y (shoulder↔finger) with sign flip
#   Bosch_Y (thumb↔pinky)     = Non-bosch_X (thumb↔pinky)
#   Bosch_Z (nail↔palm)       = Non-bosch_Z (nail↔palm)
ORIENTATION_MATRIX = np.array([
    [ 0, -1,  0],  # Bosch_X = -Non-bosch_Y (flip Y to get shoulder→finger as positive→negative)
    [ 1,  0,  0],  # Bosch_Y = +Non-bosch_X (pinky is positive for both)
    [ 0,  0,  1],  # Bosch_Z = +Non-bosch_Z (same orientation)
])

# LEGACY: Original transformation matrix from least squares regression
# This was computed without accounting for the orientation difference
# Now superseded by ORIENTATION_MATRIX + scale correction
TRANSFORM_MATRIX_LEGACY = np.array([
    [-0.680553, -3.208964, -0.447696],
    [+2.166274, +0.479881, -0.110818],
    [+0.337926, -0.280969, -2.802281],
])

# Scale correction: Non-bosch (16384 LSB/g) → Bosch (4096 LSB/g)
SCALE_FACTOR = 4096.0 / 16384.0  # = 0.25

# Offset vector: bias correction after rotation/scaling
TRANSFORM_OFFSET = np.array([-768.84, -1065.12, 429.37])

# Calibration metadata
CALIBRATION_INFO = {
    'method': 'least_squares_regression',
    'num_samples': 9991,
    'num_sessions': 4,
    'r2_x': 0.6701,
    'r2_y': 0.3888,
    'r2_z': 0.5222,
    'magnitude_correlation': 0.4605,
    'rmse_x': 1683.9,
    'rmse_y': 1392.9,
    'rmse_z': 1484.0,
    'date': '2026-01-20',
}


# =============================================================================
# TRANSFORMATION FUNCTIONS
# =============================================================================

def transform_non_bosch_to_bosch(
    acc_x: Union[float, np.ndarray],
    acc_y: Union[float, np.ndarray],
    acc_z: Union[float, np.ndarray],
    use_orientation_correction: bool = True
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Transform non_bosch accelerometer values to bosch-equivalent values.

    This applies orientation correction to account for the 90° rotation between sensors,
    plus scale correction and bias offset.

    Args:
        acc_x: Non-bosch X-axis value(s) in raw sensor units (thumb↔pinky)
        acc_y: Non-bosch Y-axis value(s) in raw sensor units (shoulder↔finger)
        acc_z: Non-bosch Z-axis value(s) in raw sensor units (nail↔palm)
        use_orientation_correction: If True, use ORIENTATION_MATRIX (recommended).
                                    If False, use legacy TRANSFORM_MATRIX_LEGACY.

    Returns:
        Tuple of (bosch_x, bosch_y, bosch_z) in bosch-equivalent units where:
            bosch_x: finger↔shoulder axis
            bosch_y: thumb↔pinky axis
            bosch_z: nail↔palm axis

    Example:
        # Single value
        bx, by, bz = transform_non_bosch_to_bosch(-500, 270, -180)

        # Batch (numpy arrays)
        bx, by, bz = transform_non_bosch_to_bosch(
            np.array([-500, -510, -505]),
            np.array([270, 275, 268]),
            np.array([-180, -175, -182])
        )
    """
    # Stack into matrix form
    non_bosch = np.array([acc_x, acc_y, acc_z])

    if use_orientation_correction:
        # Step 1: Apply orientation correction (handles 90° rotation)
        # Step 2: Apply scale factor (16384 → 4096 LSB/g)
        if non_bosch.ndim == 1:
            # Single sample: (3,)
            bosch = (ORIENTATION_MATRIX @ non_bosch) * SCALE_FACTOR
        else:
            # Batch: (3, N)
            bosch = (ORIENTATION_MATRIX @ non_bosch) * SCALE_FACTOR
    else:
        # Legacy transformation (for backwards compatibility)
        if non_bosch.ndim == 1:
            bosch = TRANSFORM_MATRIX_LEGACY @ non_bosch + TRANSFORM_OFFSET
        else:
            bosch = TRANSFORM_MATRIX_LEGACY @ non_bosch + TRANSFORM_OFFSET.reshape(3, 1)

    return bosch[0], bosch[1], bosch[2]


def transform_acc_array(acc_data: np.ndarray, use_orientation_correction: bool = True) -> np.ndarray:
    """
    Transform accelerometer data array from non_bosch to bosch-equivalent values.

    Args:
        acc_data: Accelerometer data with shape (3, N) where:
                  - acc_data[0] = Non-bosch X values (thumb↔pinky)
                  - acc_data[1] = Non-bosch Y values (shoulder↔finger)
                  - acc_data[2] = Non-bosch Z values (nail↔palm)
        use_orientation_correction: If True, use ORIENTATION_MATRIX (recommended).

    Returns:
        Transformed array with same shape (3, N) containing bosch-equivalent values:
                  - result[0] = Bosch X values (finger↔shoulder)
                  - result[1] = Bosch Y values (thumb↔pinky)
                  - result[2] = Bosch Z values (nail↔palm)

    Example:
        acc_data = np.array([
            [x1, x2, x3, ...],  # Non-bosch X axis
            [y1, y2, y3, ...],  # Non-bosch Y axis
            [z1, z2, z3, ...],  # Non-bosch Z axis
        ])
        transformed = transform_acc_array(acc_data)
    """
    if acc_data.shape[0] != 3:
        raise ValueError(f"Expected shape (3, N), got {acc_data.shape}")

    if use_orientation_correction:
        # Apply orientation correction + scale factor
        transformed = (ORIENTATION_MATRIX @ acc_data) * SCALE_FACTOR
    else:
        # Legacy transformation
        transformed = TRANSFORM_MATRIX_LEGACY @ acc_data + TRANSFORM_OFFSET.reshape(3, 1)

    return transformed


class SensorCalibrator:
    """
    Sensor calibration class for transforming non_bosch to bosch-equivalent values.

    This class wraps the transformation functions and provides additional
    functionality like calibration info and batch processing.

    Attributes:
        transform_matrix: The 3x3 rotation/scaling matrix
        transform_offset: The 3x1 bias offset vector
        calibration_info: Dictionary with calibration metadata

    Example:
        calibrator = SensorCalibrator()

        # Transform single sample
        bx, by, bz = calibrator.transform_single(-500, 270, -180)

        # Transform batch
        acc_calibrated = calibrator.transform(acc_data)  # (3, N) -> (3, N)

        # Get calibration info
        print(f"R² values: {calibrator.calibration_info}")
    """

    def __init__(self, use_orientation_correction: bool = True):
        """Initialize calibrator with pre-computed transformation parameters.

        Args:
            use_orientation_correction: If True, use ORIENTATION_MATRIX (recommended).
                                       If False, use legacy TRANSFORM_MATRIX_LEGACY.
        """
        self.use_orientation_correction = use_orientation_correction
        self.orientation_matrix = ORIENTATION_MATRIX.copy()
        self.scale_factor = SCALE_FACTOR
        self.transform_matrix_legacy = TRANSFORM_MATRIX_LEGACY.copy()
        self.transform_offset = TRANSFORM_OFFSET.copy()
        self.calibration_info = CALIBRATION_INFO.copy()

    def transform(self, acc_data: np.ndarray) -> np.ndarray:
        """
        Transform accelerometer data array.

        Args:
            acc_data: Shape (3, N) array of [X, Y, Z] values from non-bosch sensor

        Returns:
            Transformed array with same shape in bosch-equivalent units
        """
        return transform_acc_array(acc_data, self.use_orientation_correction)

    def transform_single(
        self,
        acc_x: float,
        acc_y: float,
        acc_z: float
    ) -> Tuple[float, float, float]:
        """
        Transform a single accelerometer reading.

        Args:
            acc_x: Non-bosch X-axis value (thumb↔pinky)
            acc_y: Non-bosch Y-axis value (shoulder↔finger)
            acc_z: Non-bosch Z-axis value (nail↔palm)

        Returns:
            Tuple of (bosch_x, bosch_y, bosch_z) where:
                bosch_x: finger↔shoulder axis
                bosch_y: thumb↔pinky axis
                bosch_z: nail↔palm axis
        """
        return transform_non_bosch_to_bosch(acc_x, acc_y, acc_z, self.use_orientation_correction)

    def get_accuracy_report(self) -> str:
        """Get a formatted string describing calibration accuracy."""
        info = self.calibration_info
        return f"""Sensor Calibration Accuracy Report
===================================
Method: {info['method']}
Calibration samples: {info['num_samples']}
Recording sessions: {info['num_sessions']}

Axis-wise R² (coefficient of determination):
  X-axis: {info['r2_x']:.4f} ({info['r2_x']*100:.1f}% variance explained)
  Y-axis: {info['r2_y']:.4f} ({info['r2_y']*100:.1f}% variance explained)
  Z-axis: {info['r2_z']:.4f} ({info['r2_z']*100:.1f}% variance explained)

Axis-wise RMSE (root mean square error):
  X-axis: {info['rmse_x']:.1f} raw units
  Y-axis: {info['rmse_y']:.1f} raw units
  Z-axis: {info['rmse_z']:.1f} raw units

Magnitude correlation: {info['magnitude_correlation']:.4f}

Note: R² values indicate moderate approximation quality.
Results may not match bosch sensor performance exactly.
"""


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_calibration_enabled() -> bool:
    """Check if sensor calibration should be applied based on settings."""
    try:
        from config.settings import ACC_SENSOR_TYPE
        return ACC_SENSOR_TYPE.lower() == 'non_bosch'
    except ImportError:
        return False


def get_calibration_info() -> dict:
    """Get calibration metadata dictionary."""
    return CALIBRATION_INFO.copy()


# Export for backwards compatibility
TRANSFORM_MATRIX = ORIENTATION_MATRIX  # Use orientation matrix by default


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    # Test the calibration
    print("Sensor Calibration Module Test")
    print("=" * 50)

    calibrator = SensorCalibrator()
    print(calibrator.get_accuracy_report())

    # Test single sample transform
    test_x, test_y, test_z = -500, 270, -180
    bx, by, bz = calibrator.transform_single(test_x, test_y, test_z)
    print(f"\nTest transformation:")
    print(f"  Input (non_bosch):  X={test_x}, Y={test_y}, Z={test_z}")
    print(f"  Output (bosch-eq):  X={bx:.1f}, Y={by:.1f}, Z={bz:.1f}")

    # Test batch transform
    test_data = np.array([
        [-500, -510, -505],
        [270, 275, 268],
        [-180, -175, -182]
    ])
    transformed = calibrator.transform(test_data)
    print(f"\nBatch transformation:")
    print(f"  Input shape: {test_data.shape}")
    print(f"  Output shape: {transformed.shape}")
