# hand-gesture-recognition-using-mediapipe
Estimate hand pose using MediaPipe (Python version).<br> This is a sample 
program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.
<br> ❗ _️**This is English Translated version of the [original repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe). All Content is translated to english along with comments and notebooks**_ ❗
<br> 
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

This repository contains the following contents.
* Sample program
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
python app.py
```

The following options can be specified when running the demo.
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：960)
* --height<br>Height at the time of camera capture (Default：540)
* --use_static_image_mode<br>Whether to use static_image_mode option for MediaPipe inference (Default：Unspecified)
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.5)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

# Directory
<pre>
│  app.py
│  keypoint_classification.ipynb
│  point_history_classification.ipynb
│  
├─model
│  ├─keypoint_classifier
│  │  │  keypoint.csv
│  │  │  keypoint_classifier.hdf5
│  │  │  keypoint_classifier.py
│  │  │  keypoint_classifier.tflite
│  │  └─ keypoint_classifier_label.csv
│  │          
│  └─point_history_classifier
│      │  point_history.csv
│      │  point_history_classifier.hdf5
│      │  point_history_classifier.py
│      │  point_history_classifier.tflite
│      └─ point_history_classifier_label.csv
│          
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a sample program for inference.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (index finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv)
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).<br>
If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Finger gesture recognition training
#### 1.Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
In the initial state, 4 types of learning data are included: stationary (class ID: 0), clockwise (class ID: 1), counterclockwise (class ID: 2), and moving (class ID: 4). <br>
If necessary, add 5 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2.Model training
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 4" and <br>modify the label of "model/point_history_classifier/point_history_classifier_label.csv" as appropriate. <br><br>

#### X.Model structure
The image of the model prepared in "[point_history_classification.ipynb](point_history_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. <br>Please change "use_lstm = False" to "True" when using (tf-nightly required (as of 2020/12/16))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Reference
* [MediaPipe](https://mediapipe.dev/)

# Author
Kazuhito Takahashi(https://twitter.com/KzhtTkhs)

# Translation and other improvements
1. Nikita Kiselov(https://github.com/kinivi)
2. Izzulhaq Mahardika(https://github.com/dikaizm)
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).

# SIGNAGE_CONTROL_GESTURE_GUIDE

# Signage Control Gesture Data Collection Guide

This guide explains how to collect training data for the 3 signage control gestures: **Swipe Left**, **Swipe Right**, and **Selfie**.

## Quick Start

### 1. Start the Application
```bash
python3 app_signage.py
```

### 2. Start Recording Mode
- Press **`M`** to enter recording mode
- You should see "MODE: Recording Swipe Gesture" on screen

### 3. Collect Data for Each Gesture

Data is recorded continuously while holding down a number key. Release the key to stop recording.

#### Class 0: Non Gesture (Background/Idle)
1. **Press and hold `0`** on your keyboard
2. You'll see "GESTURE: Non Gesture" on screen with "REC" indicator
3. While holding the key:
   - Keep your hand stationary or make random non-gesture movements
   - Move your hand around without performing any specific gesture
   - This helps the model distinguish gestures from normal hand movement
4. **Release `0`** to stop recording
5. **Repeat** for 100-200 samples with variations

#### Class 1: Swipe Left
1. **Press and hold `1`** on your keyboard
2. You'll see "GESTURE: Swipe Left" on screen with "REC" indicator
3. While holding the key, perform the swipe left gesture:
   - Point with your index finger (make a pointing gesture)
   - Move your hand smoothly from **right to left**
   - Keep the movement steady and consistent
4. **Release `1`** to stop recording
5. **Repeat** for 100-200 times with variations (different speeds, positions, angles)

#### Class 2: Swipe Right
1. **Press and hold `2`** on your keyboard
2. You'll see "GESTURE: Swipe Right" on screen with "REC" indicator
3. While holding the key, perform the swipe right gesture:
   - Point with your index finger
   - Move your hand smoothly from **left to right**
   - Keep the movement steady and consistent
4. **Release `2`** to stop recording
5. **Repeat** for 100-200 times with variations

### 4. Exit
- Press **`ESC`** to exit the application

---

## Data Collection Best Practices

### Recommended Sample Count
- **Minimum**: 100 samples per gesture
- **Recommended**: 200-300 samples per gesture
- **More is better**: The more varied data you collect, the better the model will perform

### Variation Guidelines

For each gesture, collect samples with:

1. **Different Speeds**
   - Fast movements
   - Slow movements
   - Medium speed

2. **Different Hand Positions**
   - Close to camera
   - Far from camera
   - Different heights (high, middle, low)

3. **Different Angles**
   - Straight on
   - Slightly tilted
   - From different sides

4. **Both Hands** (if applicable)
   - Left hand
   - Right hand

5. **Different Lighting Conditions**
   - Bright lighting
   - Dim lighting
   - Different backgrounds

### Tips for Quality Data

✅ **DO:**
- Keep your hand clearly visible
- Perform gestures smoothly and naturally
- Maintain consistency in the gesture pattern
- Collect data in the environment where the system will be used

❌ **DON'T:**
- Rush through data collection
- Block the camera with your hand
- Make jerky or inconsistent movements
- Collect all samples in exactly the same way

---

## Verifying Your Data

### Check the CSV File
```bash
# View first 5 rows
head -n 5 model/point_history_classifier/signage_control_gesture.csv

# Count samples per gesture
grep "^0," model/point_history_classifier/signage_control_gesture.csv | wc -l  # Swipe Left
grep "^1," model/point_history_classifier/signage_control_gesture.csv | wc -l  # Swipe Right
grep "^2," model/point_history_classifier/signage_control_gesture.csv | wc -l  # Selfie
```

Each row should have:
- **Label** (0, 1, or 2)
- **32 values** (16 frames × 2 coordinates per frame)

---

## Training the Model

Once you've collected enough data, you can train the model using the existing Jupyter notebook:

### Option 1: Use Existing Notebook (Recommended)
1. Open `point_history_classification.ipynb`
2. Modify the notebook to:
   - Load data from `signage_control_gesture.csv`
   - Use labels from `signage_control_gesture_label.csv`
   - Save the trained model as `signage_control_gesture_classifier.tflite`

### Option 2: Create New Notebook
Create a copy of `point_history_classification.ipynb` and adapt it for your custom gestures.

### Training Steps
1. **Load the data**: Read from `signage_control_gesture.csv`
2. **Split data**: 80% training, 20% validation
3. **Train the model**: Use the same architecture as the point history classifier
4. **Evaluate**: Check accuracy and confusion matrix
5. **Export**: Save as TFLite model

---

## Using the Trained Model

After training, you'll need to modify `app.py` to:
1. Load your custom model instead of the default point history classifier
2. Use your custom labels
3. Map the gesture predictions to signage control actions

---

## Troubleshooting

### "No data is being saved"
- Make sure you're in mode 3 (press `c`)
- Make sure you're pressing 0, 1, or 2 (not other numbers)
- Check that `signage_control_gesture.csv` exists in `model/point_history_classifier/`

### "Gestures are not being detected"
- Make sure your hand is clearly visible
- Ensure good lighting
- Keep your hand within the camera frame
- Perform gestures smoothly

### "Model accuracy is low"
- Collect more training data (aim for 200+ samples per gesture)
- Add more variation to your training data
- Check for mislabeled data in the CSV file
- Ensure gestures are distinct and consistent