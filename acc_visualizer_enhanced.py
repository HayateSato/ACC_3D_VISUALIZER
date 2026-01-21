"""
Enhanced 3D Accelerometer Visualizer with Detailed Hand Model

This version has been calibrated based on actual wearable sensor orientation:
- Bosch sensor mounted on upper side of LEFT wrist (watch position)
- X-axis: Finger-to-shoulder direction (negative = toward fingers, positive = toward shoulder)
- Y-axis: Pinky-to-thumb direction (positive = toward pinky, negative = toward thumb)
- Z-axis: Through hand (negative = toward nails/back of hand, positive = toward palm/thigh)

Features:
- Realistic left hand model with watch/wearable, thumb, and nails
- Trajectory trail showing recent motion path
- Acceleration vector visualization
- Properly oriented based on sensor mounting

Usage:
    python acc_visualizer_enhanced.py [csv_path]

If no path provided, you'll be prompted to enter one.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import sys


class LeftHandModel:
    """
    Left hand model oriented for wrist-worn accelerometer visualization.

    Sensor coordinate system (Bosch on LEFT wrist, watch position):
    - X: Finger-to-shoulder direction (negative = toward fingers, positive = toward shoulder)
    - Y: Pinky-to-thumb direction (positive = toward pinky, negative = toward thumb)
    - Z: Through hand (negative = toward nails/back, positive = toward palm/thigh)

    Hand model coordinate system (for rendering):
    - Model X: Toward pinky (left side of hand)
    - Model Y: Toward fingertips
    - Model Z: Up through back of hand

    The watch/wearable is on the upper side of the wrist.
    """

    def __init__(self):
        # Hand dimensions (in arbitrary units, scaled for visualization)
        self.palm_width = 0.8
        self.palm_length = 0.9
        self.palm_thickness = 0.15
        self.wrist_width = 0.5
        self.wrist_length = 0.5

    def create_mesh(self, rotation_matrix):
        """Create hand mesh with given rotation."""
        meshes = []

        # === WRIST (closer to body, negative Y) ===
        wrist = self._box(0, -0.5, 0, 0.5, 0.5, 0.2)
        wrist = self._rotate(wrist, rotation_matrix)
        meshes.append(('wrist', wrist, (0.88, 0.72, 0.62, 0.95)))

        # === WATCH/WEARABLE on upper wrist ===
        # Watch body (on top of wrist, Z positive)
        watch_body = self._box(0, -0.45, 0.18, 0.35, 0.4, 0.08)
        watch_body = self._rotate(watch_body, rotation_matrix)
        meshes.append(('watch_body', watch_body, (0.2, 0.2, 0.25, 1.0)))  # Dark gray

        # Watch screen
        watch_screen = self._box(0, -0.45, 0.23, 0.28, 0.32, 0.02)
        watch_screen = self._rotate(watch_screen, rotation_matrix)
        meshes.append(('watch_screen', watch_screen, (0.1, 0.3, 0.5, 1.0)))  # Blue screen

        # Watch band - upper part
        band_upper = self._box(0, -0.45, 0.12, 0.2, 0.5, 0.05)
        band_upper = self._rotate(band_upper, rotation_matrix)
        meshes.append(('band_upper', band_upper, (0.3, 0.3, 0.35, 1.0)))

        # === PALM ===
        palm = self._box(0, 0.35, 0, 0.8, 0.8, 0.15)
        palm = self._rotate(palm, rotation_matrix)
        meshes.append(('palm', palm, (0.96, 0.80, 0.69, 0.95)))

        # === FINGERS ===
        # Finger positions (X offset from center), lengths
        # For left hand: pinky on left (negative X), thumb on right (positive X)
        finger_specs = [
            (-0.32, 0.45, 'pinky'),    # Pinky (left)
            (-0.16, 0.55, 'ring'),     # Ring
            (0.0, 0.6, 'middle'),      # Middle
            (0.16, 0.55, 'index'),     # Index
        ]

        for x_off, length, name in finger_specs:
            # Each finger has 3 segments
            y_start = 0.75

            # Proximal (base)
            seg1 = self._box(x_off, y_start + 0.12, 0, 0.12, 0.22, 0.12)
            seg1 = self._rotate(seg1, rotation_matrix)
            meshes.append((f'{name}_1', seg1, (0.94, 0.78, 0.67, 0.95)))

            # Middle
            seg2 = self._box(x_off, y_start + 0.32, 0, 0.11, 0.18, 0.11)
            seg2 = self._rotate(seg2, rotation_matrix)
            meshes.append((f'{name}_2', seg2, (0.94, 0.78, 0.67, 0.95)))

            # Distal (tip)
            seg3 = self._box(x_off, y_start + 0.48, 0, 0.10, 0.14, 0.10)
            seg3 = self._rotate(seg3, rotation_matrix)
            meshes.append((f'{name}_3', seg3, (0.94, 0.78, 0.67, 0.95)))

            # Nail (on top, Z positive)
            nail = self._box(x_off, y_start + 0.52, 0.06, 0.08, 0.10, 0.02)
            nail = self._rotate(nail, rotation_matrix)
            meshes.append((f'{name}_nail', nail, (0.98, 0.9, 0.88, 1.0)))

        # === THUMB (on right side for left hand) ===
        # Thumb points outward and slightly up
        thumb_base = self._box(0.42, 0.15, 0.02, 0.15, 0.2, 0.14)
        thumb_base = self._rotate(thumb_base, rotation_matrix)
        meshes.append(('thumb_1', thumb_base, (0.94, 0.78, 0.67, 0.95)))

        thumb_mid = self._box(0.52, 0.30, 0.02, 0.13, 0.18, 0.12)
        thumb_mid = self._rotate(thumb_mid, rotation_matrix)
        meshes.append(('thumb_2', thumb_mid, (0.94, 0.78, 0.67, 0.95)))

        thumb_tip = self._box(0.58, 0.44, 0.02, 0.11, 0.14, 0.10)
        thumb_tip = self._rotate(thumb_tip, rotation_matrix)
        meshes.append(('thumb_3', thumb_tip, (0.94, 0.78, 0.67, 0.95)))

        thumb_nail = self._box(0.60, 0.48, 0.08, 0.08, 0.10, 0.02)
        thumb_nail = self._rotate(thumb_nail, rotation_matrix)
        meshes.append(('thumb_nail', thumb_nail, (0.98, 0.9, 0.88, 1.0)))

        return meshes

    def _box(self, cx, cy, cz, w, d, h):
        """Create box vertices. w=width(X), d=depth(Y), h=height(Z)"""
        return np.array([
            [cx - w/2, cy - d/2, cz - h/2],
            [cx + w/2, cy - d/2, cz - h/2],
            [cx + w/2, cy + d/2, cz - h/2],
            [cx - w/2, cy + d/2, cz - h/2],
            [cx - w/2, cy - d/2, cz + h/2],
            [cx + w/2, cy - d/2, cz + h/2],
            [cx + w/2, cy + d/2, cz + h/2],
            [cx - w/2, cy + d/2, cz + h/2],
        ])

    def _rotate(self, vertices, R):
        """Apply rotation matrix to vertices."""
        return (R @ vertices.T).T

    def _faces(self, v):
        """Get faces for a box."""
        return [
            [v[0], v[1], v[2], v[3]],  # bottom
            [v[4], v[5], v[6], v[7]],  # top
            [v[0], v[1], v[5], v[4]],  # front
            [v[2], v[3], v[7], v[6]],  # back
            [v[0], v[3], v[7], v[4]],  # left
            [v[1], v[2], v[6], v[5]],  # right
        ]


def acc_to_rotation(acc_x, acc_y, acc_z, sensor_scale=4096.0):
    """
    Convert accelerometer values to rotation matrix.

    Sensor axes (Bosch on LEFT wrist, watch position):
    - Sensor X: Finger-to-shoulder (negative = fingers, positive = shoulder)
    - Sensor Y: Pinky-to-thumb (positive = pinky, negative = thumb)
    - Sensor Z: Through hand (negative = nails/back, positive = palm/thigh)

    Hand model axes (for rendering):
    - Model X: Toward pinky (left)
    - Model Y: Toward fingertips
    - Model Z: Up through back of hand

    Mapping from sensor to model:
    - Model X (pinky) = Sensor Y (positive Y = pinky direction)
    - Model Y (fingers) = -Sensor X (negative X = finger direction)
    - Model Z (back/up) = -Sensor Z (negative Z = nails/back direction)

    Args:
        acc_x, acc_y, acc_z: Accelerometer values (sensor coordinates)
        sensor_scale: Value representing 1g (4096 for Bosch, 16384 for non_bosch)

    Returns:
        3x3 rotation matrix
    """
    # Normalize to g
    sx = acc_x / sensor_scale  # sensor X
    sy = acc_y / sensor_scale  # sensor Y
    sz = acc_z / sensor_scale  # sensor Z

    # Map sensor axes to model axes (gravity vector in model space)
    # Model coords: X=pinky, Y=fingers, Z=back of hand (up)
    gx = sy   # Model X (pinky) from Sensor Y (positive Y = pinky)
    gy = -sx  # Model Y (fingers) from -Sensor X
    gz = -sz  # Model Z (back/nails up) from -Sensor Z

    # The accelerometer measures gravity + acceleration
    # When stationary, it measures gravity pointing "down"
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    if mag < 0.1:
        return np.eye(3)

    # Normalize gravity vector in model space
    gx, gy, gz = gx/mag, gy/mag, gz/mag

    # When hand is resting with nails up (back of hand facing up):
    # - Gravity points down in world space
    # - In model space, gravity should be -Z (since model Z is "up through back")
    # - So gz should be negative when nails are up

    # Calculate pitch (rotation around model X axis - tilting fingers up/down)
    # and roll (rotation around model Y axis - rotating wrist palm up/down)

    # Pitch: how much the fingers point up or down
    # When gy is positive, fingers are pointing down (gravity toward fingers)
    pitch = np.arcsin(np.clip(-gy, -1, 1))

    # Roll: wrist rotation (palm up vs nails up)
    # When gz is negative, nails are up; when positive, palm is up
    roll = np.arctan2(gx, -gz)

    # Build rotation matrix
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # Rotation around X (pitch - tilting fingers up/down)
    Rx = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ])

    # Rotation around Y (roll - wrist rotation)
    Ry = np.array([
        [cr, 0, sr],
        [0, 1, 0],
        [-sr, 0, cr]
    ])

    return Ry @ Rx


class EnhancedVisualizer:
    """Enhanced visualizer with trajectory and detailed hand model."""

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.hand = LeftHandModel()

        # State
        self.current_sensor = 'bosch'
        self.current_frame = 0
        self.is_playing = False
        self.play_speed = 1.0
        self.show_trail = True
        self.show_acc_vector = True
        self.trail_length = 50  # Number of points in trail

        # Load data
        self._load_data()

        # Trail history
        self.trail_points = []

    def _load_data(self):
        """Load CSV data."""
        print(f"Loading: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        if self.df.columns[-1].startswith('Unnamed'):
            self.df = self.df.iloc[:, :-1]

        print(f"  Total rows: {len(self.df)}")

        # Extract sensor data
        self.acc_data = {}

        # Bosch
        if 'is_accelerometer_bosch' in self.df.columns:
            bosch_df = self.df[self.df['is_accelerometer_bosch'] == 1].copy()
            if len(bosch_df) > 0:
                self.acc_data['bosch'] = {
                    'timestamps': bosch_df['timestamp'].values,
                    'x': bosch_df['bosch_acc_x'].values,
                    'y': bosch_df['bosch_acc_y'].values,
                    'z': bosch_df['bosch_acc_z'].values,
                    'scale': 4096.0,  # Bosch BMA400 scale
                }
                print(f"  Bosch samples: {len(bosch_df)}")

        # Non-Bosch
        if 'is_accelerometer' in self.df.columns:
            nb_df = self.df[self.df['is_accelerometer'] == 1].copy()
            if len(nb_df) > 0:
                self.acc_data['non_bosch'] = {
                    'timestamps': nb_df['timestamp'].values,
                    'x': nb_df['acc_x'].values,
                    'y': nb_df['acc_y'].values,
                    'z': nb_df['acc_z'].values,
                    'scale': 16384.0,  # Typical IMU scale
                }
                print(f"  Non-Bosch samples: {len(nb_df)}")

        if not self.acc_data:
            raise ValueError("No accelerometer data found!")

        # Set default
        self.current_sensor = 'bosch' if 'bosch' in self.acc_data else list(self.acc_data.keys())[0]
        self._update_sensor_data()

    def _update_sensor_data(self):
        """Update current arrays for selected sensor."""
        data = self.acc_data[self.current_sensor]
        self.timestamps = data['timestamps']
        self.acc_x = data['x']
        self.acc_y = data['y']
        self.acc_z = data['z']
        self.sensor_scale = data['scale']
        self.n_frames = len(self.timestamps)
        self.time_seconds = (self.timestamps - self.timestamps[0]) / 1000.0
        self.acc_mag = np.sqrt(self.acc_x**2 + self.acc_y**2 + self.acc_z**2)

        # Reset trail
        self.trail_points = []

    def run(self):
        """Run visualization."""
        # Create figure
        self.fig = plt.figure(figsize=(18, 10))
        self.fig.patch.set_facecolor('#f5f5f5')

        # Grid layout
        gs = self.fig.add_gridspec(3, 3, width_ratios=[1.4, 1, 0.35],
                                   height_ratios=[1, 1, 0.12],
                                   hspace=0.25, wspace=0.15)

        # 3D view
        self.ax_3d = self.fig.add_subplot(gs[:2, 0], projection='3d')
        self._setup_3d()

        # Time series
        self.ax_xyz = self.fig.add_subplot(gs[0, 1])
        self._setup_xyz_plot()

        # Magnitude
        self.ax_mag = self.fig.add_subplot(gs[1, 1])
        self._setup_mag_plot()

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[:2, 2])
        self._setup_info_panel()

        # Controls
        self._setup_controls()

        # Title
        self.fig.suptitle(f'Left Wrist Accelerometer - {self.csv_path.name}',
                         fontsize=12, fontweight='bold')

        # Initial frame
        self._update_frame(0)

        plt.show()

    def _setup_3d(self):
        """Setup 3D axes."""
        self.ax_3d.set_xlim(-2, 2)
        self.ax_3d.set_ylim(-2, 2)
        self.ax_3d.set_zlim(-2, 2)

        # Labels with orientation hints (model coordinates for visualization)
        self.ax_3d.set_xlabel('X (← Pinky)', fontsize=9)
        self.ax_3d.set_ylabel('Y (Fingers →)', fontsize=9)
        self.ax_3d.set_zlabel('Z (↑ Back of hand)', fontsize=9)

        self.ax_3d.set_title(f'{self.current_sensor.upper()} Sensor', fontsize=11)

        # Initial view: looking at back of hand from slight angle
        self.ax_3d.view_init(elev=25, azim=-60)

        # Draw reference floor (ground plane at Z=0)
        xx, yy = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-2, 2, 5))
        self.ax_3d.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

        # Reference axes
        axis_len = 1.5
        self.ax_3d.quiver(0, 0, 0, axis_len, 0, 0, color='red', alpha=0.6, arrow_length_ratio=0.1)
        self.ax_3d.quiver(0, 0, 0, 0, axis_len, 0, color='green', alpha=0.6, arrow_length_ratio=0.1)
        self.ax_3d.quiver(0, 0, 0, 0, 0, axis_len, color='blue', alpha=0.6, arrow_length_ratio=0.1)

        self.ax_3d.text(axis_len + 0.1, 0, 0, 'X', color='red', fontsize=9)
        self.ax_3d.text(0, axis_len + 0.1, 0, 'Y', color='green', fontsize=9)
        self.ax_3d.text(0, 0, axis_len + 0.1, 'Z', color='blue', fontsize=9)

        # Gravity indicator
        self.gravity_text = self.ax_3d.text(-1.8, -1.8, -1.8, '↓ Gravity', fontsize=8, color='purple')

    def _setup_xyz_plot(self):
        """Setup XYZ time series plot."""
        self.ax_xyz.set_title('Accelerometer X, Y, Z', fontsize=10)
        self.ax_xyz.set_xlabel('Time (s)', fontsize=9)
        self.ax_xyz.set_ylabel('Value', fontsize=9)
        self.ax_xyz.grid(True, alpha=0.3)

        self.line_x, = self.ax_xyz.plot([], [], 'r-', label='X (finger↔shoulder)', lw=0.8, alpha=0.8)
        self.line_y, = self.ax_xyz.plot([], [], 'g-', label='Y (pinky↔thumb)', lw=0.8, alpha=0.8)
        self.line_z, = self.ax_xyz.plot([], [], 'b-', label='Z (nails↔palm)', lw=0.8, alpha=0.8)
        self.marker_xyz = self.ax_xyz.axvline(x=0, color='black', lw=1.5, ls='--')

        self.ax_xyz.legend(loc='upper right', fontsize=7)

        # Plot data
        self.line_x.set_data(self.time_seconds, self.acc_x)
        self.line_y.set_data(self.time_seconds, self.acc_y)
        self.line_z.set_data(self.time_seconds, self.acc_z)
        self.ax_xyz.set_xlim(0, self.time_seconds[-1])
        self.ax_xyz.relim()
        self.ax_xyz.autoscale_view()

    def _setup_mag_plot(self):
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

    def _setup_info_panel(self):
        """Setup info panel."""
        self.ax_info.axis('off')
        self.ax_info.set_xlim(0, 1)
        self.ax_info.set_ylim(0, 1)

        self.ax_info.text(0.5, 0.95, 'Current Values', fontsize=11, fontweight='bold',
                         ha='center', transform=self.ax_info.transAxes)

        # Value displays (sensor coordinates)
        self.info_labels = {}
        items = [
            ('Time:', 0.88),
            ('X (fing↔shldr):', 0.78),
            ('Y (pnky↔thmb):', 0.68),
            ('Z (nail↔palm):', 0.58),
            ('Magnitude:', 0.45),
            ('G-force:', 0.35),
            ('Orientation:', 0.22),
        ]

        for label, y in items:
            self.ax_info.text(0.05, y, label, fontsize=9, transform=self.ax_info.transAxes)
            self.info_labels[label] = self.ax_info.text(
                0.95, y, '--', fontsize=9, fontweight='bold',
                ha='right', transform=self.ax_info.transAxes
            )

        # Legend
        self.ax_info.text(0.5, 0.08, 'Watch is on\nupper wrist', fontsize=8,
                         ha='center', va='center', style='italic',
                         transform=self.ax_info.transAxes, color='gray')

    def _setup_controls(self):
        """Setup control widgets."""
        # Time slider
        ax_slider = plt.axes([0.12, 0.05, 0.5, 0.025])
        self.slider = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        # Play/Pause
        ax_play = plt.axes([0.65, 0.05, 0.06, 0.025])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._on_play)

        # Reset
        ax_reset = plt.axes([0.72, 0.05, 0.06, 0.025])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset)

        # Speed
        ax_speed = plt.axes([0.12, 0.02, 0.2, 0.02])
        self.speed_slider = Slider(ax_speed, 'Speed', 0.1, 5.0, valinit=1.0)
        self.speed_slider.on_changed(lambda v: setattr(self, 'play_speed', v))

        # Sensor selector
        if len(self.acc_data) > 1:
            ax_radio = plt.axes([0.80, 0.02, 0.12, 0.06])
            ax_radio.set_facecolor('#f5f5f5')
            labels = list(self.acc_data.keys())
            self.radio = RadioButtons(ax_radio, labels, active=labels.index(self.current_sensor))
            self.radio.on_clicked(self._on_sensor)

        # Options
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
        self.trail_points = []
        self.slider.set_val(0)

    def _on_sensor(self, label):
        self.current_sensor = label
        self._update_sensor_data()

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
            self.show_acc_vector = not self.show_acc_vector
        self._update_frame(self.current_frame)

    def _update_frame(self, idx):
        """Update visualization for frame."""
        if idx >= self.n_frames:
            idx = self.n_frames - 1

        ax, ay, az = self.acc_x[idx], self.acc_y[idx], self.acc_z[idx]
        t = self.time_seconds[idx]
        mag = self.acc_mag[idx]
        g_force = mag / self.sensor_scale

        # Calculate rotation
        R = acc_to_rotation(ax, ay, az, self.sensor_scale)

        # Clear 3D (keep only first few items: floor and axes)
        while len(self.ax_3d.collections) > 1:
            self.ax_3d.collections[-1].remove()
        while len(self.ax_3d.lines) > 3:
            self.ax_3d.lines[-1].remove()

        # Draw hand
        meshes = self.hand.create_mesh(R)
        for name, verts, color in meshes:
            faces = self.hand._faces(verts)
            poly = Poly3DCollection(faces, facecolor=color[:3], edgecolor='gray',
                                    alpha=color[3] if len(color) > 3 else 0.9,
                                    linewidth=0.3)
            self.ax_3d.add_collection3d(poly)

        # Trail
        wrist_pos = R @ np.array([0, -0.5, 0.15])  # Watch position
        self.trail_points.append(wrist_pos.copy())
        if len(self.trail_points) > self.trail_length:
            self.trail_points.pop(0)

        if self.show_trail and len(self.trail_points) > 1:
            trail = np.array(self.trail_points)
            colors = plt.cm.viridis(np.linspace(0.3, 1, len(trail) - 1))
            for i in range(len(trail) - 1):
                self.ax_3d.plot3D([trail[i, 0], trail[i+1, 0]],
                                 [trail[i, 1], trail[i+1, 1]],
                                 [trail[i, 2], trail[i+1, 2]],
                                 color=colors[i], linewidth=2.5, alpha=0.8)

        # Acceleration vector
        if self.show_acc_vector:
            vec = np.array([ax, ay, az]) / self.sensor_scale * 1.2
            self.ax_3d.quiver(wrist_pos[0], wrist_pos[1], wrist_pos[2],
                             vec[0], vec[1], vec[2],
                             color='purple', linewidth=2.5, arrow_length_ratio=0.15)

        # Update time markers
        self.marker_xyz.set_xdata([t, t])
        self.marker_mag.set_xdata([t, t])

        # Update info panel
        self.info_labels['Time:'].set_text(f'{t:.2f} s')
        self.info_labels['X (fing↔shldr):'].set_text(f'{ax:>7.0f}')
        self.info_labels['Y (pnky↔thmb):'].set_text(f'{ay:>7.0f}')
        self.info_labels['Z (nail↔palm):'].set_text(f'{az:>7.0f}')
        self.info_labels['Magnitude:'].set_text(f'{mag:>7.0f}')
        self.info_labels['G-force:'].set_text(f'{g_force:.2f} g')

        # Orientation description based on corrected axes
        # Z: negative = nails/back up, positive = palm up
        # X: negative = fingers down, positive = shoulder down
        if az < -self.sensor_scale * 0.7:
            orient = "Nails up ↑"
        elif az > self.sensor_scale * 0.7:
            orient = "Palm up ↑"
        elif ax < -self.sensor_scale * 0.7:
            orient = "Fingers down ↓"
        elif ax > self.sensor_scale * 0.7:
            orient = "Fingers up ↑"
        else:
            orient = "Tilted"
        self.info_labels['Orientation:'].set_text(orient)

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
            self.trail_points = []

        self.slider.set_val(self.current_frame)
        self.fig.canvas.get_tk_widget().after(25, self._animate)


def main():
    print("=" * 60)
    print("3D Accelerometer Visualizer - Calibrated Orientation")
    print("=" * 60)
    print()
    print("Hand model: LEFT wrist with watch on upper side")
    print("Sensor axes (Bosch):")
    print("  X: Finger ↔ Shoulder (negative = toward fingers)")
    print("  Y: Pinky ↔ Thumb (positive = toward pinky)")
    print("  Z: Nails ↔ Palm (negative = toward nails/back, positive = toward palm)")
    print()

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        print("Enter CSV file path:")
        csv_path = input("Path: ").strip().strip('"').strip("'")

    if not csv_path or not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)

    print()
    print("Controls:")
    print("  - Slider: Navigate through time")
    print("  - Play/Pause: Animate the visualization")
    print("  - Speed: Adjust animation speed")
    print("  - Trail: Toggle motion trail")
    print("  - Vector: Toggle acceleration vector")
    print("  - Drag on 3D view to rotate camera")
    print()

    viz = EnhancedVisualizer(csv_path)
    viz.run()


if __name__ == "__main__":
    main()
