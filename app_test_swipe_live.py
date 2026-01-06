"""
Live camera feed to test swipe gesture recognition model
Uses the new MediaPipe Tasks API (compatible with mediapipe 0.10.9+)
"""
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import math

# Configuration
MODEL_PATH = 'model/point_history_classifier/swipe_gesture_classifier_20260106_055250.tflite'
HAND_LANDMARKER_PATH = 'model/hand_landmarker.task'
TIME_STEPS = 16
FEATURES_PER_STEP = 16  # [x, y, dx, dy, angle, dtheta] + 5 fingertips × 2 coords

# Palm + orientation utilities for swipe gestures (must match app_signage.py)
PALM_IDS = [0, 5, 9, 13, 17]  # wrist + MCP joints
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips


def calc_palm_center(landmark_list):
    """Calculate palm centroid from landmarks in pixel coordinates."""
    xs = [landmark_list[i][0] for i in PALM_IDS]
    ys = [landmark_list[i][1] for i in PALM_IDS]
    return np.mean(xs), np.mean(ys)


def calc_fingertip_features(landmark_list, palm_x, palm_y, scale):
    """Calculate normalized fingertip positions relative to palm center."""
    features = []
    for tip_id in FINGERTIP_IDS:
        tip_x = (landmark_list[tip_id][0] - palm_x) / scale
        tip_y = (landmark_list[tip_id][1] - palm_y) / scale
        features.extend([tip_x, tip_y])
    return features  # 10 features: 5 fingertips × 2 coords


def calc_hand_orientation(landmark_list):
    """Calculate hand orientation angle from wrist to middle MCP."""
    vx = landmark_list[9][0] - landmark_list[0][0]
    vy = landmark_list[9][1] - landmark_list[0][1]
    return np.arctan2(vy, vx)


def calc_hand_scale(landmark_list):
    """Calculate hand scale using index MCP to pinky MCP distance."""
    x1, y1 = landmark_list[5]
    x2, y2 = landmark_list[17]
    return np.hypot(x2 - x1, y2 - y1) + 1e-6


# Gesture labels
GESTURE_LABELS = {
    0: 'Non-gesture',
    1: 'Swipe Left',
    2: 'Swipe Right'
}

# Colors for visualization (BGR)
COLORS = {
    0: (200, 200, 200),  # Gray for non-gesture
    1: (0, 255, 0),      # Green for swipe left
    2: (0, 0, 255)       # Red for swipe right
}


class SwipeGestureDetector:
    def __init__(self, model_path, hand_landmarker_path):
        # Load TFLite model for gesture classification
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Point history for tracking (feature space - for model input)
        self.point_history = deque(maxlen=TIME_STEPS)
        
        # Palm pixel history (pixel space - for visualization)
        self.palm_pixel_history = deque(maxlen=TIME_STEPS)
        
        # Previous values for calculating deltas
        self.prev_point = None
        self.prev_angle = None
        
        # Initialize MediaPipe Hand Landmarker (new Tasks API)
        base_options = python.BaseOptions(model_asset_path=hand_landmarker_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Define hand connections for drawing (21 landmarks, 20 connections)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Palm
        ]
        
        print(f"Model loaded successfully!")
        print(f"Input shape: {self.input_details[0]['shape']}")
        print(f"Output shape: {self.output_details[0]['shape']}")
    
    def calculate_angle(self, dx, dy):
        """Calculate angle in radians"""
        return math.atan2(dy, dx)
    
    def process_hand_landmark(self, landmark, img_width, img_height):
        """Process hand landmark using palm centroid (matching app_signage.py training data)"""
        # Convert landmarks to pixel coordinates list format
        landmark_list = []
        for lm in landmark:
            lm_x = int(lm.x * img_width)
            lm_y = int(lm.y * img_height)
            landmark_list.append([lm_x, lm_y])
        
        # Calculate palm center (matching app_signage.py)
        palm_x_px, palm_y_px = calc_palm_center(landmark_list)
        
        # Calculate hand orientation and scale (matching app_signage.py)
        angle = calc_hand_orientation(landmark_list)
        scale = calc_hand_scale(landmark_list)
        
        # Normalize palm position by hand scale (matching app_signage.py)
        palm_x_norm = palm_x_px / scale
        palm_y_norm = palm_y_px / scale
        
        # Calculate velocity in normalized feature space
        if self.prev_point is not None:
            dx = palm_x_norm - self.prev_point[0]
            dy = palm_y_norm - self.prev_point[1]
            
            # Calculate angle delta
            if self.prev_angle is not None:
                dtheta = angle - self.prev_angle
                # Normalize angle difference to [-pi, pi]
                dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            else:
                dtheta = 0.0
            
            self.prev_angle = angle
        else:
            dx, dy, dtheta = 0.0, 0.0, 0.0
        
        self.prev_point = (palm_x_norm, palm_y_norm)
        
        # Fingertip features (normalized relative to palm)
        fingertip_features = calc_fingertip_features(
            landmark_list, palm_x_px, palm_y_px, scale
        )
        
        # Feature vector: [x_norm, y_norm, dx, dy, angle, dtheta, tip1_x, tip1_y, ...]
        feature_vector = [palm_x_norm, palm_y_norm, dx, dy, angle, dtheta] + fingertip_features
        self.point_history.append(feature_vector)
        
        # Store pixel position for visualization trail
        self.palm_pixel_history.append([int(palm_x_px), int(palm_y_px)])
        
        return (int(palm_x_px), int(palm_y_px))
    
    def predict_gesture(self, confidence_threshold=0.7, motion_threshold=1.5):
        """Predict gesture from point history with direction validation"""
        if len(self.point_history) < TIME_STEPS:
            return 0, [1.0, 0.0, 0.0]  # Not enough history
        
        # Prepare input
        features = np.array(list(self.point_history), dtype=np.float32)
        
        # Calculate total horizontal displacement for direction check
        # dx is at index 2 of each feature vector [x, y, dx, dy, angle, dtheta]
        dx_values = features[:, 2]
        dy_values = features[:, 3]
        total_dx = np.sum(dx_values)
        total_dy = np.sum(dy_values)
        
        # Calculate motion magnitude - must have significant horizontal movement
        motion_magnitude = abs(total_dx)
        
        # ========== MOTION THRESHOLD CHECK ==========
        # If hand is not moving enough horizontally, it's NOT a swipe
        if motion_magnitude < motion_threshold:
            # Return non-gesture with high confidence
            return 0, np.array([0.9, 0.05, 0.05])
        
        features = features.reshape(1, TIME_STEPS, FEATURES_PER_STEP)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], features)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        predicted_class = output[0].argmax()
        probabilities = output[0]
        
        # Confidence check
        if probabilities[predicted_class] < confidence_threshold:
            return 0, probabilities  # Not confident enough
        
        # Direction sanity check for swipe gestures
        # Swipe Left = hand moves left→right = positive dx
        # Swipe Right = hand moves right→left = negative dx
        if predicted_class == 1 and total_dx < 0:  # Swipe Left should have positive dx
            return 0, probabilities  # Direction mismatch
        if predicted_class == 2 and total_dx > 0:  # Swipe Right should have negative dx
            return 0, probabilities  # Direction mismatch
        
        return predicted_class, probabilities
    
    def reset_history(self):
        """Reset point history when hand is lost"""
        self.point_history.clear()
        self.palm_pixel_history.clear()
        self.prev_point = None
        self.prev_angle = None
    
    def detect_hands(self, frame, timestamp_ms):
        """Detect hands using the new MediaPipe Tasks API"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        return result
    
    def draw_landmarks(self, frame, hand_landmarks_list):
        """Draw hand landmarks on frame"""
        for hand_landmarks in hand_landmarks_list:
            # Draw landmarks manually
            h, w, _ = frame.shape
            landmark_points = []
            
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw connections
            for connection in self.hand_connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmark_points) and end_idx < len(landmark_points):
                    cv2.line(frame, landmark_points[start_idx], landmark_points[end_idx], (0, 255, 0), 2)
    
    def close(self):
        """Clean up resources"""
        self.hand_landmarker.close()


def main():
    # Initialize detector
    detector = SwipeGestureDetector(MODEL_PATH, HAND_LANDMARKER_PATH)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\n=== Swipe Gesture Recognition ===")
    print("Controls:")
    print("  - Move your palm left/right to perform swipe gestures")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset history")
    print("\n")
    
    current_gesture = 0
    gesture_probs = [1.0, 0.0, 0.0]
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Calculate timestamp in milliseconds
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if timestamp_ms == 0:
            timestamp_ms = int(frame_count * (1000 / 30))  # Assume 30 FPS
        frame_count += 1
        
        # Detect hands using new API
        results = detector.detect_hands(frame, timestamp_ms)
        
        # Process hand detection
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                # Draw hand landmarks
                detector.draw_landmarks(frame, [hand_landmarks])
                
                # Process landmark and get position
                pos = detector.process_hand_landmark(hand_landmarks, w, h)
                
                # Draw palm center
                cv2.circle(frame, pos, 10, (255, 0, 255), -1)
                
                # Predict gesture
                current_gesture, gesture_probs = detector.predict_gesture()
        else:
            # No hand detected, reset history
            detector.reset_history()
            current_gesture = 0
            gesture_probs = [1.0, 0.0, 0.0]
        
        # Draw trail of palm center history (using pixel-space positions)
        if len(detector.palm_pixel_history) > 1:
            points = list(detector.palm_pixel_history)
            
            # Draw trail
            for i in range(len(points) - 1):
                if points[i][0] != 0 and points[i][1] != 0:  # Skip zero positions
                    alpha = (i + 1) / len(points)
                    thickness = int(alpha * 3) + 1
                    cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), (255, 255, 0), thickness)
        
        # Calculate and display motion info for debugging
        motion_dx = 0.0
        motion_mag = 0.0
        if len(detector.point_history) >= TIME_STEPS:
            features = np.array(list(detector.point_history))
            motion_dx = np.sum(features[:, 2])  # Sum of dx values
            motion_mag = abs(motion_dx)
        
        # Display gesture information
        gesture_name = GESTURE_LABELS[current_gesture]
        color = COLORS[current_gesture]
        
        # Background for text
        cv2.rectangle(frame, (10, 10), (500, 200), (0, 0, 0), -1)
        
        # Gesture label
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Motion indicator - show if above threshold
        motion_threshold = 1.5
        is_moving = motion_mag > motion_threshold
        motion_color = (0, 255, 0) if is_moving else (100, 100, 100)
        cv2.putText(frame, f"Motion: {motion_mag:.2f} (threshold: {motion_threshold})", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)
        cv2.putText(frame, f"Direction dx: {motion_dx:.2f}", (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Probabilities
        y_offset = 135
        for i, (label, prob) in enumerate(zip(GESTURE_LABELS.values(), gesture_probs)):
            text = f"{label}: {prob:.2%}"
            cv2.putText(frame, text, (20, y_offset + i * 22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # History buffer status
        buffer_text = f"Buffer: {len(detector.point_history)}/{TIME_STEPS}"
        cv2.putText(frame, buffer_text, (w - 250, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit | 'r' to reset", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Swipe Gesture Recognition', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_history()
            print("History reset!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
