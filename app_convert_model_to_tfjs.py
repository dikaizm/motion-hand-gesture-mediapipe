"""
Convert Keras 3 model (.keras) to TensorFlow.js LayersModel format.
Automatically fixes Keras 3 → TFJS key incompatibilities.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import json
import keras
import tensorflowjs as tfjs

# Paths
INPUT_MODEL = 'model/point_history_classifier/swipe_gesture_classifier_20260106_055250.keras'
OUTPUT_DIR = '../frontend/public/models/swipe_gesture_tfjs'

print(f"Loading Keras model from: {INPUT_MODEL}")
model = keras.models.load_model(INPUT_MODEL)

print("Model summary:")
model.summary()

print(f"\nConverting to TensorFlow.js format...")
print(f"Output directory: {OUTPUT_DIR}")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert to TFJS LayersModel format
tfjs.converters.save_keras_model(model, OUTPUT_DIR)

# Fix Keras 3 → TFJS key incompatibilities
model_json_path = os.path.join(OUTPUT_DIR, 'model.json')
print(f"\nPatching model.json for TFJS compatibility...")

with open(model_json_path, 'r') as f:
    model_json = json.load(f)

# Replace Keras 3 keys with TFJS-compatible keys
model_json_str = json.dumps(model_json)
model_json_str = model_json_str.replace('"batch_shape"', '"batchInputShape"')
model_json_str = model_json_str.replace('"build_input_shape"', '"buildInputShape"')
model_json = json.loads(model_json_str)

with open(model_json_path, 'w') as f:
    json.dump(model_json, f)

print("✅ Patched: batch_shape → batchInputShape")
print("✅ Patched: build_input_shape → buildInputShape")

print("\n✅ Conversion complete!")
print(f"Model files saved to: {OUTPUT_DIR}")
