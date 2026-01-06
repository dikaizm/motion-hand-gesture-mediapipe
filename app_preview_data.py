#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Live Preview Tool for Gesture Data
===================================
Visualizes swipe gesture data as a moving dot on a 2D plane.

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
        self.label_map = {0: 'Swipe Left', 1: 'Swipe Right'}
        
        # Parse data to find appropriate axis limits
        self.parse_data_bounds()
        
        # Setup Plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()
        
        # Animation - 20 FPS
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=50, blit=False, cache_frame_data=False
        )
        
        # Event handling
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        print("\n=== Swipe Data Player ===")
        print(f"Loaded {len(self.df)} samples")
        print("Controls: ← → navigate | Space pause | Q quit")
        plt.show()

    def parse_data_bounds(self):
        """Find min/max of x, y across all data for axis limits."""
        all_x = []
        all_y = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            features = row[1:].values.reshape(-1, 6)  # 16 steps × 6 features
            all_x.extend(features[:, 0])
            all_y.extend(features[:, 1])
        
        self.x_min, self.x_max = min(all_x), max(all_x)
        self.y_min, self.y_max = min(all_y), max(all_y)
        
        # Add some padding
        x_pad = (self.x_max - self.x_min) * 0.1
        y_pad = (self.y_max - self.y_min) * 0.1
        self.x_min -= x_pad
        self.x_max += x_pad
        self.y_min -= y_pad
        self.y_max += y_pad

    def setup_plot(self):
        self.ax.set_xlim(self.x_min, self.x_max)
        self.ax.set_ylim(self.y_min, self.y_max)
        self.ax.set_xlabel('X (normalized palm position)')
        self.ax.set_ylabel('Y (normalized palm position)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='box')
        
        # Main point (current position)
        self.point, = self.ax.plot([], [], 'o', color='blue', markersize=15, zorder=10)
        
        # Trail (history)
        self.trail, = self.ax.plot([], [], '-o', color='cyan', alpha=0.5, linewidth=2, markersize=4, zorder=5)
        
        # Start marker
        self.start_marker, = self.ax.plot([], [], 's', color='green', markersize=10, label='Start', zorder=8)
        
        # Info Text
        self.title_text = self.ax.set_title("")
        self.info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, 
                                       verticalalignment='top', fontsize=10, 
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def get_current_sample_data(self):
        row = self.df.iloc[self.current_sample_idx]
        label = int(row[0])
        # Each row: label + 16 frames × 6 features = 97 columns
        time_series = row[1:].values.reshape(-1, 6)
        return label, time_series

    def update(self, _):
        label, data = self.get_current_sample_data()
        num_frames = len(data)
        
        # Update title
        label_name = self.label_map.get(label, f"Label {label}")
        color = 'blue' if label == 0 else 'red'
        self.ax.set_title(f"Sample {self.current_sample_idx + 1}/{len(self.df)} | {label_name} | Frame {self.current_frame + 1}/{num_frames}", 
                          fontsize=12, fontweight='bold')
        
        # Features: [palm_x, palm_y, dx, dy, angle, dtheta]
        current = data[self.current_frame]
        x, y = current[0], current[1]
        dx, dy = current[2], current[3]
        angle_deg = np.degrees(current[4])
        
        # Update main point
        self.point.set_data([x], [y])
        self.point.set_color(color)
        
        # Update trail (all frames up to current)
        trail_xs = data[:self.current_frame + 1, 0]
        trail_ys = data[:self.current_frame + 1, 1]
        self.trail.set_data(trail_xs, trail_ys)
        
        # Update start marker
        self.start_marker.set_data([data[0, 0]], [data[0, 1]])
        
        # Update info text
        info = (
            f"Position: ({x:.2f}, {y:.2f})\n"
            f"Velocity: ({dx:.3f}, {dy:.3f})\n"
            f"Angle: {angle_deg:.1f}°"
        )
        self.info_text.set_text(info)
        
        # Advance frame if playing
        if self.is_playing:
            self.current_frame = (self.current_frame + 1) % num_frames
        
        return self.point, self.trail, self.start_marker, self.info_text

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
        
        print(f"Loaded: {os.path.basename(args.file)}")
        print(f"Samples: {len(df)}, Features per sample: {num_cols - 1}")
        
        # Swipe data: 1 + 16*6 = 97 columns
        if num_cols > 50:
            player = SwipeDataPlayer(df)
        else:
            print(f"Unsupported format: {num_cols} columns")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
