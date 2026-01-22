"""
3D Body Accelerometer Visualizer - Full Body Animation from Wrist Sensor

This visualizer shows a full human body animation based on wrist accelerometer data.
Since we only have a single wrist sensor, the body animation uses template-based
motion patterns while the hand/wrist orientation is driven by actual sensor data.

Activity Detection:
- STATIC: Low acceleration variance, person is stationary (hand may still move)
- WALKING: Rhythmic acceleration patterns typical of arm swing during walking
- FALLING: High acceleration spike followed by impact pattern

Animation Approach:
- Body motion is driven by pre-defined templates based on detected activity
- Hand/wrist orientation is driven by actual sensor data
- This hybrid approach provides meaningful visualization despite limited sensor data

Usage:
    python acc_body_visualizer.py [csv_path]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
import sys
from collections import deque

# Try to import calibration and config from local directory
try:
    from sensor_calibration import transform_non_bosch_to_bosch
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("Note: Sensor calibration module not found. Non-bosch sensors will use raw values.")

# Import sensor configuration
try:
    from sensor_config import BOSCH_CONFIG, NON_BOSCH_CONFIG, detect_data_format
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    BOSCH_CONFIG = {'lsb_per_g': 4096.0, 'sample_rate_hz': 25}
    NON_BOSCH_CONFIG = {'lsb_per_g': 16384.0, 'sample_rate_hz': 100}


# =============================================================================
# ACTIVITY DETECTION
# =============================================================================

class ActivityDetector:
    """
    Detect activity type from accelerometer data using sliding window analysis.

    Activities:
    - STATIC: Standing/sitting still (hand may move independently)
    - WALKING: Rhythmic arm swing during walking
    - FALLING: High-g impact event
    """

    STATIC = 'static'
    WALKING = 'walking'
    FALLING = 'falling'

    def __init__(self, sensor_scale=4096.0, window_size=25):
        self.sensor_scale = sensor_scale
        self.window_size = window_size
        self.acc_history = deque(maxlen=window_size)
        self.mag_history = deque(maxlen=window_size)

        # Thresholds (in g units)
        self.fall_threshold = 2.5  # g - high impact
        self.walk_variance_min = 0.05  # Minimum variance for walking
        self.walk_variance_max = 0.8   # Maximum variance for walking
        self.static_variance_max = 0.03  # Below this is static

    def update(self, acc_x, acc_y, acc_z):
        """Add new sample and return detected activity."""
        # Convert to g
        gx = acc_x / self.sensor_scale
        gy = acc_y / self.sensor_scale
        gz = acc_z / self.sensor_scale
        mag = np.sqrt(gx**2 + gy**2 + gz**2)

        self.acc_history.append((gx, gy, gz))
        self.mag_history.append(mag)

        if len(self.mag_history) < self.window_size // 2:
            return self.STATIC, 0.0

        # Check for fall (high g-force)
        if mag > self.fall_threshold:
            return self.FALLING, mag

        # Calculate variance over window
        mags = np.array(self.mag_history)
        variance = np.var(mags)

        # Classify based on variance
        if variance < self.static_variance_max:
            return self.STATIC, variance
        elif variance < self.walk_variance_max:
            return self.WALKING, variance
        else:
            # High variance could be falling or vigorous activity
            if max(mags) > self.fall_threshold:
                return self.FALLING, max(mags)
            return self.WALKING, variance

    def get_walk_phase(self, time_seconds):
        """Get walking animation phase (0 to 2*pi) based on time."""
        walk_frequency = 1.8  # ~1.8 Hz typical walking cadence
        return (time_seconds * walk_frequency * 2 * np.pi) % (2 * np.pi)


# =============================================================================
# HUMAN BODY MODEL (Stick Figure)
# =============================================================================

class HumanBodyModel:
    """
    Stick figure human body model with articulated joints.

    Joint hierarchy:
    - hip_center (root)
      - spine -> chest -> neck -> head
      - left_hip -> left_knee -> left_ankle -> left_foot
      - right_hip -> right_knee -> right_ankle -> right_foot
      - left_shoulder -> left_elbow -> left_wrist -> left_hand
      - right_shoulder -> right_elbow -> right_wrist -> right_hand

    Coordinate system:
    - X: Left-Right (positive = right)
    - Y: Forward-Backward (positive = forward)
    - Z: Up-Down (positive = up)
    """

    # Joint names
    JOINTS = [
        'hip_center', 'spine', 'chest', 'neck', 'head',
        'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',
        'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand',
        'left_hip', 'left_knee', 'left_ankle', 'left_foot',
        'right_hip', 'right_knee', 'right_ankle', 'right_foot'
    ]

    # Bone connections (parent, child)
    BONES = [
        ('hip_center', 'spine'), ('spine', 'chest'), ('chest', 'neck'), ('neck', 'head'),
        ('chest', 'left_shoulder'), ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'), ('left_wrist', 'left_hand'),
        ('chest', 'right_shoulder'), ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'), ('right_wrist', 'right_hand'),
        ('hip_center', 'left_hip'), ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'), ('left_ankle', 'left_foot'),
        ('hip_center', 'right_hip'), ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'), ('right_ankle', 'right_foot')
    ]

    def __init__(self):
        # Body proportions (relative to total height = 2.0 units)
        self.height = 2.0

        # Default T-pose positions
        self.t_pose = self._create_t_pose()

    def _create_t_pose(self):
        """Create default T-pose joint positions."""
        h = self.height
        positions = {
            'hip_center': np.array([0, 0, h * 0.5]),
            'spine': np.array([0, 0, h * 0.58]),
            'chest': np.array([0, 0, h * 0.72]),
            'neck': np.array([0, 0, h * 0.82]),
            'head': np.array([0, 0, h * 0.95]),

            'left_shoulder': np.array([-0.22, 0, h * 0.78]),
            'left_elbow': np.array([-0.45, 0, h * 0.62]),
            'left_wrist': np.array([-0.65, 0, h * 0.48]),
            'left_hand': np.array([-0.72, 0, h * 0.45]),

            'right_shoulder': np.array([0.22, 0, h * 0.78]),
            'right_elbow': np.array([0.45, 0, h * 0.62]),
            'right_wrist': np.array([0.65, 0, h * 0.48]),
            'right_hand': np.array([0.72, 0, h * 0.45]),

            'left_hip': np.array([-0.12, 0, h * 0.48]),
            'left_knee': np.array([-0.12, 0, h * 0.26]),
            'left_ankle': np.array([-0.12, 0, h * 0.04]),
            'left_foot': np.array([-0.12, 0.08, h * 0.02]),

            'right_hip': np.array([0.12, 0, h * 0.48]),
            'right_knee': np.array([0.12, 0, h * 0.26]),
            'right_ankle': np.array([0.12, 0, h * 0.04]),
            'right_foot': np.array([0.12, 0.08, h * 0.02]),
        }
        return positions

    def get_standing_pose(self):
        """Return relaxed standing pose with arms at sides."""
        pose = {}
        for joint, pos in self.t_pose.items():
            pose[joint] = pos.copy()

        # Adjust arms to hang naturally at sides
        h = self.height
        pose['left_shoulder'] = np.array([-0.18, 0, h * 0.76])
        pose['left_elbow'] = np.array([-0.20, 0.05, h * 0.58])
        pose['left_wrist'] = np.array([-0.18, 0.08, h * 0.42])
        pose['left_hand'] = np.array([-0.16, 0.10, h * 0.38])

        pose['right_shoulder'] = np.array([0.18, 0, h * 0.76])
        pose['right_elbow'] = np.array([0.20, 0.05, h * 0.58])
        pose['right_wrist'] = np.array([0.18, 0.08, h * 0.42])
        pose['right_hand'] = np.array([0.16, 0.10, h * 0.38])

        return pose

    def get_walking_pose(self, phase):
        """
        Return walking pose at given phase (0 to 2*pi).

        Walking involves:
        - Alternating leg swing
        - Counter arm swing (opposite arm forward when leg forward)
        - Slight torso rotation
        - Vertical bounce
        """
        pose = self.get_standing_pose()
        h = self.height

        # Leg swing amplitude
        leg_swing = 0.15
        arm_swing = 0.12

        # Vertical bounce (double frequency - bounce on each step)
        bounce = 0.02 * np.sin(2 * phase)

        # Apply bounce to upper body
        for joint in ['hip_center', 'spine', 'chest', 'neck', 'head',
                     'left_shoulder', 'right_shoulder']:
            pose[joint][2] += bounce

        # Left leg (forward when phase = 0)
        leg_phase = np.sin(phase)
        pose['left_hip'][1] += leg_swing * 0.3 * leg_phase
        pose['left_knee'][1] += leg_swing * leg_phase
        pose['left_knee'][2] += 0.05 * (1 - np.cos(phase))  # Knee lift
        pose['left_ankle'][1] += leg_swing * 1.2 * leg_phase
        pose['left_foot'][1] += leg_swing * 1.2 * leg_phase

        # Right leg (opposite phase)
        pose['right_hip'][1] -= leg_swing * 0.3 * leg_phase
        pose['right_knee'][1] -= leg_swing * leg_phase
        pose['right_knee'][2] += 0.05 * (1 + np.cos(phase))  # Knee lift
        pose['right_ankle'][1] -= leg_swing * 1.2 * leg_phase
        pose['right_foot'][1] -= leg_swing * 1.2 * leg_phase

        # Right arm swings forward when left leg forward (counter-swing)
        arm_phase = np.sin(phase)
        pose['right_elbow'][1] += arm_swing * arm_phase
        pose['right_wrist'][1] += arm_swing * 1.3 * arm_phase
        pose['right_hand'][1] += arm_swing * 1.5 * arm_phase

        # Left arm swings opposite
        pose['left_elbow'][1] -= arm_swing * arm_phase
        pose['left_wrist'][1] -= arm_swing * 1.3 * arm_phase
        pose['left_hand'][1] -= arm_swing * 1.5 * arm_phase

        return pose

    def get_falling_pose(self, fall_progress):
        """
        Return falling pose based on fall progress (0 = start, 1 = on ground).

        Fall sequence (forward fall - whole body tips forward):
        - 0.0-0.2: Loss of balance, body starts tipping forward from ankles
        - 0.2-0.6: Body falling forward as a unit, knees bend, arms reach out
        - 0.6-0.85: Hands hit ground, body continues down
        - 0.85-1.0: Final position on ground
        """
        h = self.height
        ground_height = 0.12

        # Final lying pose (for interpolation target)
        final_pose = {
            'hip_center': np.array([0, 0.1, ground_height + 0.08]),
            'spine': np.array([0, 0.25, ground_height + 0.10]),
            'chest': np.array([0, 0.42, ground_height + 0.12]),
            'neck': np.array([0, 0.55, ground_height + 0.10]),
            'head': np.array([0, 0.70, ground_height + 0.08]),
            'left_shoulder': np.array([-0.18, 0.45, ground_height + 0.10]),
            'left_elbow': np.array([-0.32, 0.58, ground_height + 0.06]),
            'left_wrist': np.array([-0.40, 0.70, ground_height + 0.04]),
            'left_hand': np.array([-0.42, 0.78, ground_height + 0.03]),
            'right_shoulder': np.array([0.18, 0.45, ground_height + 0.10]),
            'right_elbow': np.array([0.32, 0.58, ground_height + 0.06]),
            'right_wrist': np.array([0.40, 0.70, ground_height + 0.04]),
            'right_hand': np.array([0.42, 0.78, ground_height + 0.03]),
            'left_hip': np.array([-0.12, -0.10, ground_height + 0.08]),
            'left_knee': np.array([-0.12, -0.38, ground_height + 0.06]),
            'left_ankle': np.array([-0.12, -0.65, ground_height + 0.05]),
            'left_foot': np.array([-0.12, -0.75, ground_height + 0.04]),
            'right_hip': np.array([0.12, -0.10, ground_height + 0.08]),
            'right_knee': np.array([0.12, -0.38, ground_height + 0.06]),
            'right_ankle': np.array([0.12, -0.65, ground_height + 0.05]),
            'right_foot': np.array([0.12, -0.75, ground_height + 0.04]),
        }

        if fall_progress >= 0.85:
            # Final lying position - just return the final pose
            return final_pose

        # Get standing pose as starting point
        pose = self.get_standing_pose()

        if fall_progress < 0.2:
            # Phase 1: Initial stumble - body tips forward from feet
            t = fall_progress / 0.2

            # Pivot around ankles - whole body tips forward
            ankle_z = pose['left_ankle'][2]
            ankle_y = pose['left_ankle'][1]

            # Small forward rotation angle (up to ~15 degrees)
            angle = t * 0.26
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            for joint in pose:
                if 'ankle' in joint or 'foot' in joint:
                    continue
                # Rotate around ankle position
                rel_y = pose[joint][1] - ankle_y
                rel_z = pose[joint][2] - ankle_z

                # Rotate forward (positive angle tips head forward)
                new_y = rel_y * cos_a + rel_z * sin_a
                new_z = -rel_y * sin_a + rel_z * cos_a

                pose[joint][1] = ankle_y + new_y
                pose[joint][2] = ankle_z + new_z

            # Arms start moving forward reflexively
            arm_fwd = t * 0.15
            pose['left_elbow'][1] += arm_fwd
            pose['left_wrist'][1] += arm_fwd * 1.5
            pose['left_hand'][1] += arm_fwd * 2
            pose['right_elbow'][1] += arm_fwd
            pose['right_wrist'][1] += arm_fwd * 1.5
            pose['right_hand'][1] += arm_fwd * 2

        elif fall_progress < 0.6:
            # Phase 2: Main fall - body tips more, starts dropping, knees bend
            t = (fall_progress - 0.2) / 0.4

            # Smooth acceleration (ease-in for gravity effect)
            t_accel = t * t  # Quadratic for acceleration feel

            # Rotation increases (15 to ~70 degrees)
            angle = 0.26 + t * 0.96
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Feet stay planted initially, then start sliding
            foot_slide = t * 0.15
            ankle_z = pose['left_ankle'][2]
            ankle_y = pose['left_ankle'][1] - foot_slide

            # Drop the whole body as it falls
            drop = t_accel * h * 0.3

            for joint in pose:
                if 'foot' in joint:
                    pose[joint][1] -= foot_slide
                    continue
                if 'ankle' in joint:
                    pose[joint][1] -= foot_slide
                    pose[joint][2] -= drop * 0.3
                    continue

                # Rotate around lowering ankle position
                rel_y = pose[joint][1] - (ankle_y + foot_slide)
                rel_z = pose[joint][2] - ankle_z

                new_y = rel_y * cos_a + rel_z * sin_a
                new_z = -rel_y * sin_a + rel_z * cos_a

                pose[joint][1] = ankle_y + new_y
                pose[joint][2] = ankle_z + new_z - drop

            # Knees bend as body falls
            knee_bend = t * 0.25
            pose['left_knee'][1] += knee_bend * 0.5
            pose['left_knee'][2] -= knee_bend * 0.3
            pose['right_knee'][1] += knee_bend * 0.5
            pose['right_knee'][2] -= knee_bend * 0.3

            # Arms reach forward to catch fall
            arm_reach = t * 0.4
            pose['left_elbow'][1] += arm_reach * 0.8
            pose['left_wrist'][1] += arm_reach * 1.2
            pose['left_wrist'][2] -= t * 0.2  # Hands going down
            pose['left_hand'][1] += arm_reach * 1.5
            pose['left_hand'][2] -= t * 0.3
            pose['right_elbow'][1] += arm_reach * 0.8
            pose['right_wrist'][1] += arm_reach * 1.2
            pose['right_wrist'][2] -= t * 0.2
            pose['right_hand'][1] += arm_reach * 1.5
            pose['right_hand'][2] -= t * 0.3

        else:
            # Phase 3: Impact transition - smoothly interpolate to final pose
            t = (fall_progress - 0.6) / 0.25  # 0.6 to 0.85

            # Use smooth interpolation (ease-out)
            t_smooth = 1 - (1 - t) * (1 - t)

            # Get the pose at end of phase 2 as starting point
            phase2_end = self.get_falling_pose(0.599)

            # Interpolate each joint from phase2_end to final_pose
            for joint in pose:
                pose[joint] = phase2_end[joint] * (1 - t_smooth) + final_pose[joint] * t_smooth

        # Ensure nothing goes below ground
        for joint in pose:
            if pose[joint][2] < ground_height:
                pose[joint][2] = ground_height

        return pose

    def apply_wrist_rotation(self, pose, rotation_matrix, side='left'):
        """
        Apply sensor-derived rotation to wrist/hand.

        Args:
            pose: Current pose dictionary
            rotation_matrix: 3x3 rotation matrix from accelerometer
            side: 'left' or 'right'
        """
        wrist_key = f'{side}_wrist'
        hand_key = f'{side}_hand'
        elbow_key = f'{side}_elbow'

        wrist_pos = pose[wrist_key]
        elbow_pos = pose[elbow_key]

        # Calculate hand offset from wrist in local coordinates
        # Then rotate it
        hand_offset = np.array([0, 0.02, -0.08])  # Default hand offset
        rotated_offset = rotation_matrix @ hand_offset

        # Scale down the rotation effect for more subtle movement
        blend_factor = 0.6
        rotated_offset = blend_factor * rotated_offset + (1 - blend_factor) * hand_offset

        pose[hand_key] = wrist_pos + rotated_offset

        return pose


# =============================================================================
# BODY VISUALIZER
# =============================================================================

class BodyVisualizer:
    """Main body visualizer class."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.body = HumanBodyModel()
        self.activity_detector = ActivityDetector()

        # State
        self.current_sensor = 'bosch'
        self.current_frame = 0
        self.is_playing = False
        self.play_speed = 1.0
        self.show_trail = True
        self.show_vector = True
        self.apply_calibration = True
        self.trail_length = 50
        self.wrist_trail = []

        # Fall animation state
        self.fall_start_frame = None
        self.fall_duration_frames = 30  # ~1 second at 25Hz

        # Load data
        self._load_data()

    def _load_data(self):
        """Load and parse CSV data."""
        print(f"Loading: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        if df.columns[-1].startswith('Unnamed'):
            df = df.iloc[:, :-1]

        self.acc_data = {}

        # Bosch sensor (baseline)
        if 'is_accelerometer_bosch' in df.columns:
            bosch_df = df[df['is_accelerometer_bosch'] == 1].copy().reset_index(drop=True)
            if len(bosch_df) > 0:
                self.acc_data['bosch'] = {
                    'timestamps': bosch_df['timestamp'].values,
                    'x': bosch_df['bosch_acc_x'].values,
                    'y': bosch_df['bosch_acc_y'].values,
                    'z': bosch_df['bosch_acc_z'].values,
                    'scale': BOSCH_CONFIG['lsb_per_g'],
                }
                print(f"  Bosch samples: {len(bosch_df)}")

        # Non-Bosch sensor
        if 'is_accelerometer' in df.columns:
            nb_df = df[df['is_accelerometer'] == 1].copy().reset_index(drop=True)
            if len(nb_df) > 0:
                x_vals = nb_df['acc_x'].values
                y_vals = nb_df['acc_y'].values
                z_vals = nb_df['acc_z'].values

                if CONFIG_AVAILABLE:
                    data_info = detect_data_format(x_vals, y_vals, z_vals, NON_BOSCH_CONFIG['lsb_per_g'])
                    print(f"  Non-Bosch data format: {data_info['format']} (magnitude: {data_info['magnitude']:.2f})")
                    detected_scale = data_info['estimated_scale']
                else:
                    detected_scale = NON_BOSCH_CONFIG['lsb_per_g']

                self.acc_data['non_bosch'] = {
                    'timestamps': nb_df['timestamp'].values,
                    'x': x_vals,
                    'y': y_vals,
                    'z': z_vals,
                    'scale': detected_scale,
                }
                print(f"  Non-Bosch samples: {len(nb_df)}")

        if not self.acc_data:
            raise ValueError("No accelerometer data found!")

        self.current_sensor = 'bosch' if 'bosch' in self.acc_data else 'non_bosch'
        self._update_sensor()

    def _update_sensor(self):
        """Update current sensor data."""
        data = self.acc_data[self.current_sensor]
        self.timestamps = data['timestamps']
        self.acc_x = data['x'].copy()
        self.acc_y = data['y'].copy()
        self.acc_z = data['z'].copy()
        self.sensor_scale = data['scale']
        self.n_frames = len(self.timestamps)
        self.time_seconds = (self.timestamps - self.timestamps[0]) / 1000.0

        # Apply calibration for non_bosch
        if self.current_sensor == 'non_bosch' and self.apply_calibration and CALIBRATION_AVAILABLE:
            print("  Applying calibration transform...")

            if self.sensor_scale > 5000:
                for i in range(len(self.acc_x)):
                    cx, cy, cz = transform_non_bosch_to_bosch(
                        self.acc_x[i], self.acc_y[i], self.acc_z[i]
                    )
                    self.acc_x[i], self.acc_y[i], self.acc_z[i] = cx, cy, cz
                self.sensor_scale = BOSCH_CONFIG['lsb_per_g']
            elif self.sensor_scale > 100:
                mg_to_bosch_scale = BOSCH_CONFIG['lsb_per_g'] / 1000.0
                for i in range(len(self.acc_x)):
                    temp_x = -self.acc_y[i] * mg_to_bosch_scale
                    temp_y = self.acc_x[i] * mg_to_bosch_scale
                    temp_z = -self.acc_z[i] * mg_to_bosch_scale
                    self.acc_x[i], self.acc_y[i], self.acc_z[i] = temp_x, temp_y, temp_z
                self.sensor_scale = BOSCH_CONFIG['lsb_per_g']
            else:
                g_to_bosch_scale = BOSCH_CONFIG['lsb_per_g']
                for i in range(len(self.acc_x)):
                    temp_x = -self.acc_y[i] * g_to_bosch_scale
                    temp_y = self.acc_x[i] * g_to_bosch_scale
                    temp_z = -self.acc_z[i] * g_to_bosch_scale
                    self.acc_x[i], self.acc_y[i], self.acc_z[i] = temp_x, temp_y, temp_z
                self.sensor_scale = BOSCH_CONFIG['lsb_per_g']

        self.acc_mag = np.sqrt(self.acc_x**2 + self.acc_y**2 + self.acc_z**2)

        # Update activity detector scale
        self.activity_detector.sensor_scale = self.sensor_scale

        # Pre-detect activities for all frames
        self._precompute_activities()

        self.wrist_trail = []
        self.fall_start_frame = None

    def _precompute_activities(self):
        """Pre-compute activity detection for smooth visualization."""
        self.activities = []
        self.activity_values = []

        detector = ActivityDetector(self.sensor_scale)

        for i in range(self.n_frames):
            activity, value = detector.update(self.acc_x[i], self.acc_y[i], self.acc_z[i])
            self.activities.append(activity)
            self.activity_values.append(value)

        print(f"  Activity detection complete:")
        print(f"    Static frames: {self.activities.count(ActivityDetector.STATIC)}")
        print(f"    Walking frames: {self.activities.count(ActivityDetector.WALKING)}")
        print(f"    Falling frames: {self.activities.count(ActivityDetector.FALLING)}")

    def _acc_to_rotation(self, acc_x, acc_y, acc_z):
        """Convert accelerometer values to rotation matrix for wrist."""
        sx = acc_x / self.sensor_scale
        sy = acc_y / self.sensor_scale
        sz = acc_z / self.sensor_scale

        gx = sy
        gy = -sx
        gz = -sz

        mag = np.sqrt(gx**2 + gy**2 + gz**2)
        if mag < 0.1:
            return np.eye(3)

        gx, gy, gz = gx/mag, gy/mag, gz/mag

        pitch = np.arcsin(np.clip(-gy, -1, 1))
        roll = np.arctan2(gx, -gz)

        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Ry = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])

        return Ry @ Rx

    def run(self):
        """Run the visualization."""
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#f5f5f5')

        # Create layout
        gs = self.fig.add_gridspec(3, 3, width_ratios=[1.4, 1, 0.35],
                                   height_ratios=[1, 1, 0.12],
                                   hspace=0.25, wspace=0.15)

        # 3D view
        self.ax_3d = self.fig.add_subplot(gs[:2, 0], projection='3d')
        self._setup_3d()

        # XYZ plot
        self.ax_xyz = self.fig.add_subplot(gs[0, 1])
        self._setup_xyz()

        # Magnitude plot
        self.ax_mag = self.fig.add_subplot(gs[1, 1])
        self._setup_mag()

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[:2, 2])
        self._setup_info()

        # Controls
        self._setup_controls()

        # Title
        self.fig.suptitle(f'Body Motion Visualizer - {self.csv_path.name}',
                         fontsize=12, fontweight='bold')

        # Initial update
        self._update_frame(0)

        plt.show()

    def _setup_3d(self):
        """Setup 3D axes."""
        self.ax_3d.set_xlim(-1.5, 1.5)
        self.ax_3d.set_ylim(-1.5, 1.5)
        self.ax_3d.set_zlim(0, 2.5)

        self.ax_3d.set_xlabel('X (Left-Right)', fontsize=9)
        self.ax_3d.set_ylabel('Y (Front-Back)', fontsize=9)
        self.ax_3d.set_zlabel('Z (Up)', fontsize=9)

        self.ax_3d.set_title(f'{self.current_sensor.upper()} Sensor', fontsize=11)

        # View from front-side angle
        self.ax_3d.view_init(elev=15, azim=-45)

        # Draw ground plane
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 5), np.linspace(-1.5, 1.5, 5))
        self.ax_3d.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.15, color='green')

    def _setup_xyz(self):
        """Setup XYZ time series."""
        self.ax_xyz.set_title('Accelerometer X, Y, Z', fontsize=10)
        self.ax_xyz.set_xlabel('Time (s)', fontsize=9)
        self.ax_xyz.set_ylabel('Value', fontsize=9)
        self.ax_xyz.grid(True, alpha=0.3)

        self.line_x, = self.ax_xyz.plot([], [], 'r-', label='X', lw=0.8, alpha=0.8)
        self.line_y, = self.ax_xyz.plot([], [], 'g-', label='Y', lw=0.8, alpha=0.8)
        self.line_z, = self.ax_xyz.plot([], [], 'b-', label='Z', lw=0.8, alpha=0.8)
        self.marker_xyz = self.ax_xyz.axvline(x=0, color='black', lw=1.5, ls='--')

        self.ax_xyz.legend(loc='upper right', fontsize=7)

        self.line_x.set_data(self.time_seconds, self.acc_x)
        self.line_y.set_data(self.time_seconds, self.acc_y)
        self.line_z.set_data(self.time_seconds, self.acc_z)
        self.ax_xyz.set_xlim(0, self.time_seconds[-1])
        self.ax_xyz.relim()
        self.ax_xyz.autoscale_view()

    def _setup_mag(self):
        """Setup magnitude plot."""
        self.ax_mag.set_title('Acceleration Magnitude', fontsize=10)
        self.ax_mag.set_xlabel('Time (s)', fontsize=9)
        self.ax_mag.set_ylabel('Magnitude', fontsize=9)
        self.ax_mag.grid(True, alpha=0.3)

        self.line_mag, = self.ax_mag.plot([], [], 'm-', lw=0.8, alpha=0.8)
        self.ax_mag.axhline(y=self.sensor_scale, color='gray', ls=':', alpha=0.5, label='~1g')
        self.marker_mag = self.ax_mag.axvline(x=0, color='black', lw=1.5, ls='--')

        self.ax_mag.legend(loc='upper right', fontsize=7)

        self.line_mag.set_data(self.time_seconds, self.acc_mag)
        self.ax_mag.set_xlim(0, self.time_seconds[-1])
        self.ax_mag.relim()
        self.ax_mag.autoscale_view()

    def _setup_info(self):
        """Setup info panel."""
        self.ax_info.axis('off')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)

        self.ax_info.text(0.5, 0.95, 'Status', fontsize=11, fontweight='bold',
                         ha='center', transform=self.ax_info.transAxes)

        self.info_labels = {}
        items = [
            ('Time:', 0.88),
            ('Activity:', 0.78),
            ('G-force:', 0.68),
            ('X:', 0.55),
            ('Y:', 0.45),
            ('Z:', 0.35),
            ('Magnitude:', 0.22),
        ]

        for label, y in items:
            self.ax_info.text(0.05, y, label, fontsize=9, transform=self.ax_info.transAxes)
            self.info_labels[label] = self.ax_info.text(
                0.95, y, '--', fontsize=9, fontweight='bold',
                ha='right', transform=self.ax_info.transAxes
            )

        # Legend
        self.ax_info.text(0.5, 0.08, 'Body animation is\ntemplate-based', fontsize=8,
                         ha='center', va='center', style='italic',
                         transform=self.ax_info.transAxes, color='gray')

    def _setup_controls(self):
        """Setup control widgets."""
        ax_slider = plt.axes([0.12, 0.05, 0.5, 0.025])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        ax_play = plt.axes([0.65, 0.05, 0.06, 0.025])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._on_play)

        ax_reset = plt.axes([0.72, 0.05, 0.06, 0.025])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        ax_speed = plt.axes([0.12, 0.02, 0.2, 0.02])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        self.speed_slider.on_changed(lambda v: setattr(self, 'play_speed', v))

        if len(self.acc_data) > 1:
            ax_radio = plt.axes([0.80, 0.02, 0.12, 0.06])
            ax_radio.set_facecolor('#f5f5f5')
            labels = list(self.acc_data.keys())
            self.radio = RadioButtons(ax_radio, labels, active=labels.index(self.current_sensor))
            self.radio.on_clicked(self._on_sensor)

        ax_check = plt.axes([0.38, 0.01, 0.18, 0.035])
        ax_check.set_facecolor('#f5f5f5')
        self.check = CheckButtons(ax_check, ['Trail', 'Vector'], [True, True])
        self.check.on_clicked(self._on_check)

    def _on_slider(self, val):
        self.current_frame = int(val)
        self._update_frame(self.current_frame)

    def _on_play(self, event):
        self.is_playing = not self.is_playing
        self.btn_play.label.set_text('Pause' if self.is_playing else 'Play')
        if self.is_playing:
            self._animate()

    def _on_reset(self, event):
        self.current_frame = 0
        self.wrist_trail = []
        self.fall_start_frame = None
        self.slider.set_val(0)

    def _on_sensor(self, label):
        self.current_sensor = label
        self._update_sensor()

        self.line_x.set_data(self.time_seconds, self.acc_x)
        self.line_y.set_data(self.time_seconds, self.acc_y)
        self.line_z.set_data(self.time_seconds, self.acc_z)
        self.line_mag.set_data(self.time_seconds, self.acc_mag)

        self.ax_xyz.relim()
        self.ax_xyz.autoscale_view()
        self.ax_xyz.set_xlim(0, self.time_seconds[-1])
        self.ax_mag.relim()
        self.ax_mag.autoscale_view()
        self.ax_mag.set_xlim(0, self.time_seconds[-1])

        self.slider.valmax = self.n_frames - 1
        self.slider.set_val(min(self.current_frame, self.n_frames - 1))
        self.ax_3d.set_title(f'{self.current_sensor.upper()} Sensor', fontsize=11)
        self._update_frame(self.current_frame)

    def _on_check(self, label):
        if label == 'Trail':
            self.show_trail = not self.show_trail
        elif label == 'Vector':
            self.show_vector = not self.show_vector
        self._update_frame(self.current_frame)

    def _draw_skeleton(self, pose, color='blue', linewidth=3):
        """Draw skeleton from pose dictionary."""
        for parent, child in self.body.BONES:
            p1 = pose[parent]
            p2 = pose[child]
            self.ax_3d.plot3D([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                            color=color, linewidth=linewidth, solid_capstyle='round')

        # Draw joint markers
        for joint, pos in pose.items():
            size = 80 if joint == 'head' else 30
            c = 'orange' if 'wrist' in joint or 'hand' in joint else color
            self.ax_3d.scatter([pos[0]], [pos[1]], [pos[2]], c=c, s=size, alpha=0.9)

    def _update_frame(self, idx):
        """Update visualization for frame."""
        if idx >= self.n_frames:
            idx = self.n_frames - 1

        ax, ay, az = self.acc_x[idx], self.acc_y[idx], self.acc_z[idx]
        t = self.time_seconds[idx]
        mag = self.acc_mag[idx]
        g_force = mag / self.sensor_scale

        activity = self.activities[idx]

        # Handle fall animation timing
        if activity == ActivityDetector.FALLING and self.fall_start_frame is None:
            self.fall_start_frame = idx
        elif activity != ActivityDetector.FALLING and self.fall_start_frame is not None:
            # Keep fall pose for a bit after detection ends
            if idx - self.fall_start_frame > self.fall_duration_frames * 2:
                self.fall_start_frame = None

        # Get body pose based on activity
        if self.fall_start_frame is not None:
            fall_progress = min(1.0, (idx - self.fall_start_frame) / self.fall_duration_frames)
            pose = self.body.get_falling_pose(fall_progress)
        elif activity == ActivityDetector.WALKING:
            walk_phase = self.activity_detector.get_walk_phase(t)
            pose = self.body.get_walking_pose(walk_phase)
        else:
            pose = self.body.get_standing_pose()

        # Apply wrist rotation from sensor
        R = self._acc_to_rotation(ax, ay, az)
        pose = self.body.apply_wrist_rotation(pose, R, side='left')

        # Clear 3D
        while len(self.ax_3d.collections) > 1:
            self.ax_3d.collections[-1].remove()
        while len(self.ax_3d.lines) > 0:
            self.ax_3d.lines[-1].remove()

        # Draw skeleton
        self._draw_skeleton(pose)

        # Wrist trail
        wrist_pos = pose['left_wrist'].copy()
        self.wrist_trail.append(wrist_pos)
        if len(self.wrist_trail) > self.trail_length:
            self.wrist_trail.pop(0)

        if self.show_trail and len(self.wrist_trail) > 1:
            trail = np.array(self.wrist_trail)
            colors = plt.cm.plasma(np.linspace(0.3, 1, len(trail) - 1))
            for i in range(len(trail) - 1):
                self.ax_3d.plot3D([trail[i, 0], trail[i+1, 0]],
                                 [trail[i, 1], trail[i+1, 1]],
                                 [trail[i, 2], trail[i+1, 2]],
                                 color=colors[i], linewidth=2, alpha=0.7)

        # Acceleration vector at wrist
        if self.show_vector:
            vec = np.array([ax, ay, az]) / self.sensor_scale * 0.3
            self.ax_3d.quiver(wrist_pos[0], wrist_pos[1], wrist_pos[2],
                             vec[0], vec[1], vec[2],
                             color='purple', linewidth=2, arrow_length_ratio=0.2)

        # Update time markers
        self.marker_xyz.set_xdata([t, t])
        self.marker_mag.set_xdata([t, t])

        # Update info panel
        self.info_labels['Time:'].set_text(f'{t:.2f} s')
        self.info_labels['Activity:'].set_text(activity.upper())
        self.info_labels['G-force:'].set_text(f'{g_force:.2f} g')
        self.info_labels['X:'].set_text(f'{ax:>7.0f}')
        self.info_labels['Y:'].set_text(f'{ay:>7.0f}')
        self.info_labels['Z:'].set_text(f'{az:>7.0f}')
        self.info_labels['Magnitude:'].set_text(f'{mag:>7.0f}')

        # Color code activity
        activity_colors = {
            ActivityDetector.STATIC: 'green',
            ActivityDetector.WALKING: 'blue',
            ActivityDetector.FALLING: 'red'
        }
        self.info_labels['Activity:'].set_color(activity_colors.get(activity, 'black'))

        # Color code G-force
        if g_force > 3:
            self.info_labels['G-force:'].set_color('red')
        elif g_force > 2:
            self.info_labels['G-force:'].set_color('orange')
        else:
            self.info_labels['G-force:'].set_color('black')

        self.fig.canvas.draw_idle()

    def _animate(self):
        if not self.is_playing:
            return

        self.current_frame += max(1, int(self.play_speed))
        if self.current_frame >= self.n_frames:
            self.current_frame = 0
            self.wrist_trail = []
            self.fall_start_frame = None

        self.slider.set_val(self.current_frame)
        self.fig.canvas.get_tk_widget().after(25, self._animate)


def main():
    print("=" * 60)
    print("Body Motion Visualizer - Full Body Animation")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Stick figure human body model")
    print("  - Activity detection: Static, Walking, Falling")
    print("  - Template-based body animation")
    print("  - Real hand orientation from wrist sensor")
    print()
    print("Note: Body motion is inferred from activity detection,")
    print("      while hand/wrist orientation comes from sensor data.")
    print()

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Enter CSV file path:")
        csv_path = input("Path: ").strip().strip('"').strip("'")

    if not csv_path or not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    viz = BodyVisualizer(csv_path)
    viz.run()


if __name__ == "__main__":
    main()
