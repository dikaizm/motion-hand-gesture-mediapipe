#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live Preview Tool for Gesture Data
===================================
Visualizes swipe gesture data as a moving dot on a 2D plane.
Shows palm center trajectory and fingertip positions.

Data format per frame (16 features):
- [palm_x, palm_y, dx, dy, angle, dtheta] (6 base features)
- [tip1_x, tip1_y, tip2_x, tip2_y, tip3_x, tip3_y, tip4_x, tip4_y, tip5_x, tip5_y] (10 fingertip features)

Total: 16 frames × 16 features = 256 features + 1 label = 257 columns

Controls:
- Left/Right Arrow: Navigate between samples
- Space: Pause/Resume animation
- Q: Quit
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import argparse
import os

# Constants matching data collection
FEATURES_PER_FRAME = 16  # 6 base + 10 fingertip
NUM_FRAMES = 16
FINGERTIP_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
FINGERTIP_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

def parse_args():
    parser = argparse.ArgumentParser(description='Preview gesture data')
    parser.add_argument('--file', '-f', required=True, help='Path to the CSV file to preview')
    return parser.parse_args()

class SwipeDataPlayer:
    """Animated player for swipe gesture time-series data."""
    
    def __init__(self, df):
        self.df = df
        self.current_sample_idx = 0
        self.current_frame = 0
        self.is_playing = True
        self.label_map = {0: 'Non-gesture', 1: 'Swipe Left', 2: 'Swipe Right'}
        
        # Parse data to find appropriate axis limits
        self.parse_data_bounds()
        
        # Setup Plot - BIGGER figure
        self.fig, self.ax = plt.subplots(figsize=(14, 12))
        self.setup_plot()
        
        # Animation - 20 FPS
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=False, cache_frame_data=False
        )
        
        # Event handling
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n=== Swipe Data Player ===")
        print(f"Loaded {len(self.df)} samples")
        print(f"Features per frame: {FEATURES_PER_FRAME}, Frames: {NUM_FRAMES}")
        print("Controls: ← → navigate | Space pause | Q quit")
        plt.show()

    def parse_data_bounds(self):
        """Find min/max of all coordinates across all data for axis limits."""
        all_x = []
        all_y = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            features = row[1:].values.reshape(NUM_FRAMES, FEATURES_PER_FRAME)
            
            # Palm positions
            all_x.extend(features[:, 0])
            all_y.extend(features[:, 1])
            
            # Fingertip positions (relative to palm, need to add palm position)
            for frame_idx in range(NUM_FRAMES):
                palm_x, palm_y = features[frame_idx, 0], features[frame_idx, 1]
                for tip_idx in range(5):
                    tip_x = features[frame_idx, 6 + tip_idx * 2]
                    tip_y = features[frame_idx, 7 + tip_idx * 2]
                    # These are relative coords, add palm to get absolute
                    all_x.append(palm_x + tip_x)
                    all_y.append(palm_y + tip_y)
        
        self.x_min, self.x_max = min(all_x), max(all_x)
        self.y_min, self.y_max = min(all_y), max(all_y)
        
        # Add padding
        x_pad = (self.x_max - self.x_min) * 0.15
        y_pad = (self.y_max - self.y_min) * 0.15
        self.x_min -= x_pad
        self.x_max += x_pad
        self.y_min -= y_pad
        self.y_max += y_pad

    def setup_plot(self):
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlabel('X (normalized position)', fontsize=12)
        self.ax.set_ylabel('Y (normalized position)', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Palm center point (current position) - BIGGER
        self.palm_point, = self.ax.plot([], [], 'o', color='blue', markersize=20, zorder=15, label='Palm Center')
        
        # Palm trail (history)
        self.palm_trail, = self.ax.plot([], [], '-o', color='cyan', alpha=0.6, linewidth=3, markersize=8, zorder=5)
        
        # Start marker
        self.start_marker, = self.ax.plot([], [], 's', color='green', markersize=14, label='Start', zorder=12)
        
        # Fingertip points - BIGGER
        self.fingertip_points = []
        for i, (name, color) in enumerate(zip(FINGERTIP_NAMES, FINGERTIP_COLORS)):
            point, = self.ax.plot([], [], 'o', color=color, markersize=14, zorder=10, label=name)
            self.fingertip_points.append(point)
        
        # Fingertip trails
        self.fingertip_trails = []
        for color in FINGERTIP_COLORS:
            trail, = self.ax.plot([], [], '-', color=color, alpha=0.3, linewidth=2, zorder=3)
            self.fingertip_trails.append(trail)
        
        # Legend
        self.ax.legend(loc='upper right', fontsize=10)
        
        # Info Text - BIGGER font
        self.title_text = self.ax.set_title("", fontsize=14, fontweight='bold')
        self.info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, 
                                       verticalalignment='top', fontsize=11, 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    def get_current_sample_data(self):
        row = self.df.iloc[self.current_sample_idx]
        label = int(row[0])
        # Each row: label + 16 frames × 16 features = 257 columns
        time_series = row[1:].values.reshape(NUM_FRAMES, FEATURES_PER_FRAME)
        return label, time_series

    def update(self, _):
        label, data = self.get_current_sample_data()
        num_frames = len(data)
        
        # Update title
        label_name = self.label_map.get(label, f"Label {label}")
        if label == 0:
            color = 'gray'
        elif label == 1:
            color = 'blue'
        else:
            color = 'red'
        self.ax.set_title(f"Sample {self.current_sample_idx + 1}/{len(self.df)} | {label_name} | Frame {self.current_frame + 1}/{num_frames}", 
                          fontsize=14, fontweight='bold')
        
        # Current frame features: [palm_x, palm_y, dx, dy, angle, dtheta, tip1_x, tip1_y, ...]
        current = data[self.current_frame]
        palm_x, palm_y = current[0], current[1]
        dx, dy = current[2], current[3]
        angle_deg = np.degrees(current[4])
        dtheta_deg = np.degrees(current[5])
        
        # Update palm point
        self.palm_point.set_data([palm_x], [palm_y])
        self.palm_point.set_color(color)
        
        # Update palm trail (all frames up to current)
        trail_xs = data[:self.current_frame + 1, 0]
        trail_ys = data[:self.current_frame + 1, 1]
        self.palm_trail.set_data(trail_xs, trail_ys)
        
        # Update start marker
        self.start_marker.set_data([data[0, 0]], [data[0, 1]])
        
        # Update fingertip positions (relative to palm)
        for tip_idx in range(5):
            tip_rel_x = current[6 + tip_idx * 2]
            tip_rel_y = current[7 + tip_idx * 2]
            # Convert relative to absolute position
            tip_abs_x = palm_x + tip_rel_x
            tip_abs_y = palm_y + tip_rel_y
            self.fingertip_points[tip_idx].set_data([tip_abs_x], [tip_abs_y])
            
            # Update fingertip trails
            tip_trail_xs = []
            tip_trail_ys = []
            for frame_idx in range(self.current_frame + 1):
                frame_palm_x = data[frame_idx, 0]
                frame_palm_y = data[frame_idx, 1]
                frame_tip_rel_x = data[frame_idx, 6 + tip_idx * 2]
                frame_tip_rel_y = data[frame_idx, 7 + tip_idx * 2]
                tip_trail_xs.append(frame_palm_x + frame_tip_rel_x)
                tip_trail_ys.append(frame_palm_y + frame_tip_rel_y)
            self.fingertip_trails[tip_idx].set_data(tip_trail_xs, tip_trail_ys)
        
        # Update info text
        velocity_mag = np.sqrt(dx**2 + dy**2)
        info = (
            f"Palm: ({palm_x:.2f}, {palm_y:.2f})\n"
            f"Velocity: ({dx:.3f}, {dy:.3f})\n"
            f"Speed: {velocity_mag:.3f}\n"
            f"Angle: {angle_deg:.1f}°\n"
            f"ΔAngle: {dtheta_deg:.1f}°"
        )
        self.info_text.set_text(info)
        
        # Advance frame if playing
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % num_frames
        
        return [self.palm_point, self.palm_trail, self.start_marker, self.info_text] + self.fingertip_points + self.fingertip_trails

    def on_key(self, event):
        if event.key == 'right':
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.df)
            self.current_frame = 0  # Reset to start of new sample
        elif event.key == 'left':
            self.current_sample_idx = (self.current_sample_idx - 1) % len(self.df)
            self.current_frame = 0
        elif event.key == ' ':
            self.is_playing = not self.is_playing
            status = "Playing" if self.is_playing else "Paused"
            print(f"[{status}]")
        elif event.key == 'q':
            plt.close(self.fig)


def main():
    args = parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found")
        return

    try:
        df = pd.read_csv(args.file, header=None)
        num_cols = df.shape[1]
        expected_cols = 1 + NUM_FRAMES * FEATURES_PER_FRAME  # 1 label + 16*16 = 257
        
        print(f"Loaded: {os.path.basename(args.file)}")
        print(f"Samples: {len(df)}, Columns: {num_cols}")
        print(f"Expected: {expected_cols} columns (1 label + {NUM_FRAMES} frames × {FEATURES_PER_FRAME} features)")
        
        # Validate format
        if num_cols == expected_cols:
            print("✓ Format matches data collection format")
            player = SwipeDataPlayer(df)
        else:
            print(f"✗ Format mismatch! Got {num_cols} columns, expected {expected_cols}")
            print("Make sure the data was collected with the current app_data_collection.py")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
