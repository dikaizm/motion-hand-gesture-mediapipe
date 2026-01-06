#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import os
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier


# Palm + orientation utilities for swipe gestures
PALM_IDS = [0, 5, 9, 13, 17]  # wrist + MCP joints
FINGERTIP_IDS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips

def calc_palm_center(landmarks):
    xs = [landmarks[i][0] for i in PALM_IDS]
    ys = [landmarks[i][1] for i in PALM_IDS]
    return np.mean(xs), np.mean(ys)

def calc_fingertip_features(landmarks, palm_x, palm_y, scale):
    """Calculate normalized fingertip positions relative to palm center."""
    features = []
    for tip_id in FINGERTIP_IDS:
        tip_x = (landmarks[tip_id][0] - palm_x) / scale
        tip_y = (landmarks[tip_id][1] - palm_y) / scale
        features.extend([tip_x, tip_y])
    return features  # 10 features: 5 fingertips × 2 coords

def calc_hand_orientation(landmarks):
    # Vector from wrist (0) to middle MCP (9)
    vx = landmarks[9][0] - landmarks[0][0]
    vy = landmarks[9][1] - landmarks[0][1]
    return np.arctan2(vy, vx)

def calc_hand_scale(landmarks):
    # Scale using index MCP to pinky MCP
    x1, y1 = landmarks[5]
    x2, y2 = landmarks[17]
    return np.hypot(x2 - x1, y2 - y1) + 1e-6


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load - MediaPipe Tasks API #############################################################
    # Get the path to the hand landmarker model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'model', 'hand_landmarker.task')
    
    # Create HandLandmarker options
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,  # Enable two-hand detection for selfie frame gesture
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    
    # Create the hand landmarker
    detector = vision.HandLandmarker.create_from_options(options)

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(os.path.join(script_dir, 'model/keypoint_classifier/keypoint_classifier_label.csv'),
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            os.path.join(script_dir, 'model/point_history_classifier/point_history_classifier_label.csv'),
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Swipe feature history (NORMALIZED - for dataset recording)
    history_length = 16
    swipe_feature_history = deque(maxlen=history_length)
    prev_palm_normalized = None  # Normalized palm position for velocity calculation
    prev_angle = None
    
    # Palm pixel history (PIXEL SPACE - for visualization only)
    palm_pixel_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    frame_timestamp_ms = 0
    
    # Session timestamp for unique data files per app run
    session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"Session started: {session_timestamp}")
    
    # Mode and recording state
    current_mode = None  # None, 'motion', or 'pose'
    held_gesture_key = -1  # Currently held key (0, 1, 2) or -1 if none
    recording_count = 0  # Count of frames recorded in current hold
    
    # CSV file handle for streaming writes
    csv_file = None
    csv_writer = None

    while True:
        fps = cvFpsCalc.get()

        # Process Key #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            if csv_file:
                csv_file.close()
            break
        
        # Mode selection
        if key == ord('m') or key == ord('M'):
            current_mode = 'motion'
            print("=== MOTION MODE ===")
            print("Hold 0=Non-gesture, 1=Swipe Left, 2=Swipe Right")
        
        # Handle press-and-hold for motion mode
        if current_mode == 'motion':
            # Press-and-hold logic: record while 0/1/2 is pressed, stop when released
            if key in [ord('0'), ord('1'), ord('2')]:
                new_held_key = key - ord('0')
                if held_gesture_key != new_held_key:
                    # New gesture key pressed - start recording
                    held_gesture_key = new_held_key
                    recording_count = 0
                    csv_path = f'model/point_history_classifier/swipe_gesture_{session_timestamp}.csv'
                    if csv_file is None:
                        csv_file = open(csv_path, 'a', newline="")
                        csv_writer = csv.writer(csv_file)
                    gesture_names = ['Non-gesture', 'Swipe Left', 'Swipe Right']
                    print(f"Recording: {gesture_names[held_gesture_key]}")
                # else: same key still held, continue recording
            elif held_gesture_key >= 0:
                # Key was released (key == -1 or different key pressed) - stop recording
                if recording_count > 0:
                    print(f"Stopped. Saved {recording_count} frames.")
                held_gesture_key = -1
                recording_count = 0

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        # Create MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect hand landmarks using VIDEO mode (requires timestamp)
        frame_timestamp_ms += 33  # Approximate 30fps increment
        detection_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

        #  ####################################################################
        if detection_result.hand_landmarks:
            # Collect all hands data
            hands_data = []
            for hand_idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                handedness = detection_result.handedness[hand_idx]
                hand_label = handedness[0].category_name  # "Left" or "Right"
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                hands_data.append({
                    'label': hand_label,
                    'landmarks': landmark_list,
                    'brect': brect,
                    'handedness': handedness,
                    'raw_landmarks': hand_landmarks
                })
                
                # Draw landmarks for all detected hands
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
            
            # Handle recording based on mode
            # === MOTION MODE (press-and-hold recording) ===
            if current_mode == 'motion' and len(hands_data) > 0:
                landmark_list = hands_data[0]['landmarks']
                
                # === PIXEL SPACE (for visualization) ===
                palm_x_px, palm_y_px = calc_palm_center(landmark_list)
                palm_pixel_history.append([int(palm_x_px), int(palm_y_px)])
                
                # === FEATURE SPACE (for dataset recording) ===
                angle = calc_hand_orientation(landmark_list)
                scale = calc_hand_scale(landmark_list)
                
                # Normalize palm position by hand scale
                palm_x_norm = palm_x_px / scale
                palm_y_norm = palm_y_px / scale
                
                # Velocity in feature space
                if prev_palm_normalized is None:
                    dx, dy = 0.0, 0.0
                    dtheta = 0.0
                else:
                    dx = palm_x_norm - prev_palm_normalized[0]
                    dy = palm_y_norm - prev_palm_normalized[1]
                    dtheta = angle - prev_angle
                    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
                
                prev_palm_normalized = (palm_x_norm, palm_y_norm)
                prev_angle = angle
                
                # Fingertip features (normalized relative to palm)
                fingertip_features = calc_fingertip_features(
                    landmark_list, palm_x_px, palm_y_px, scale
                )
                
                # Feature vector: [palm_x, palm_y, dx, dy, angle, dtheta, tip1_x, tip1_y, ...]
                swipe_feature = [palm_x_norm, palm_y_norm, dx, dy, angle, dtheta] + fingertip_features
                swipe_feature_history.append(swipe_feature)
                
                # === CONTINUOUS RECORDING (while key is held) ===
                if held_gesture_key >= 0 and len(swipe_feature_history) == history_length:
                    frame_features = list(itertools.chain.from_iterable(swipe_feature_history))
                    # Write directly to CSV
                    if csv_writer:
                        csv_writer.writerow([held_gesture_key, *frame_features])
                        csv_file.flush()  # Ensure data is written
                        recording_count += 1
                
                # For display - classify hand sign
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    hands_data[0]['brect'],
                    hands_data[0]['handedness'],
                    keypoint_classifier_labels[hand_sign_id],
                    "",
                )
            elif current_mode == 'motion':
                # No hand detected
                prev_palm_normalized = None
                prev_angle = None
                palm_pixel_history.append([0, 0])
            else:
                # No mode selected - just track for visualization
                if len(hands_data) > 0:
                    palm_x_px, palm_y_px = calc_palm_center(hands_data[0]['landmarks'])
                    palm_pixel_history.append([int(palm_x_px), int(palm_y_px)])
                else:
                    palm_pixel_history.append([0, 0])
        else:
            prev_palm_normalized = None
            prev_angle = None
            palm_pixel_history.append([0, 0])

        # Draw palm center trail (PIXEL SPACE - correct visualization)
        debug_image = draw_palm_pixel_history(debug_image, palm_pixel_history)
        # Draw info overlay with current mode and recording status
        debug_image = draw_info_motion_mode(debug_image, fps, current_mode, held_gesture_key, recording_count)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    detector.close()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    """Calculate bounding rect from MediaPipe Tasks landmarks (normalized coordinates)"""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    """Calculate landmark list from MediaPipe Tasks landmarks (normalized coordinates)"""
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list



def save_gesture_sample(gesture_label, recorded_point_history, window_size, session_timestamp):
    """
    Save gesture samples to separate CSV files based on gesture type.
    Each session creates unique files with timestamp.
    
    Swipe gestures (label 0, 1) -> swipe_gesture_YYYYMMDD_HHMMSS.csv
    - TIME-SERIES: Each row is ONE frame's observation (16-step history)
    - Row format: [x, y, dx, dy, angle, d_angle] * 16 (96 features)
    - A single 30-frame recording generates 30 rows of data.
    
    Selfie gesture (label 2) -> selfie_gesture_YYYYMMDD_HHMMSS.csv
    - SINGLE FRAME: Each row is ONE selfie capture
    - Row format: [left_thumb_x, y, ... right_index_y] (8 features)
    """
    if len(recorded_point_history) == 0:
        return
    
    # Determine which CSV file to save to based on gesture type
    if gesture_label in [0, 1]:  # Swipe gestures (time-series)
        csv_path = f'model/point_history_classifier/swipe_gesture_{session_timestamp}.csv'
        save_label = gesture_label  # 0 = Swipe Left, 1 = Swipe Right
        
        # Slicing ensures we don't save more than the window size if buffer overshot
        frames_to_save = recorded_point_history[:window_size]
        
        # Write EACH frame as a separate training sample
        # frame_data is already a flattened list of 16 history steps (96 features)
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            for frame_data in frames_to_save:
                writer.writerow([save_label, *frame_data])
                
    else:  # Selfie gesture (label 2) - SINGLE FRAME
        csv_path = f'model/point_history_classifier/selfie_gesture_{session_timestamp}.csv'
        save_label = 0  # Remap to 0 for selfie model
        
        # Single frame - just take the first (and only) frame
        frame_data = recorded_point_history[0]
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([save_label, *frame_data])


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    # Get handedness label from MediaPipe Tasks format
    info_text = handedness[0].category_name
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_palm_pixel_history(image, palm_pixel_history):
    """Draw palm center trail using PIXEL SPACE coordinates for correct visualization."""
    for index, point in enumerate(palm_pixel_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info_motion_mode(image, fps, current_mode, held_gesture_key, recording_count):
    """Draw info overlay for motion mode recording."""
    # FPS
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    gesture_names = ['Non-gesture', 'Swipe Left', 'Swipe Right']
    
    # Mode indicator
    if current_mode == 'motion':
        cv.putText(image, "MODE: MOTION", (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(image, "Hold 0/1/2 to record", (10, 95),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        
        # Recording status
        if held_gesture_key >= 0:
            # Currently recording
            cv.putText(image, "REC", (10, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(image, gesture_names[held_gesture_key], (70, 130),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            cv.putText(image, f"Frames: {recording_count}", (10, 160),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    else:
        cv.putText(image, "Press M for Motion Mode", (10, 70),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    return image


if __name__ == '__main__':
    main()
