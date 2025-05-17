import os
print(f"[INFO] Current working directory: {os.getcwd()}")
print("[INFO] Attempting to import tensorflow...")
from tensorflow.keras.models import load_model
print("[INFO] TensorFlow imported.")
print("[INFO] Attempting to import mediapipe...")
import mediapipe as mp
print("[INFO] MediaPipe imported.")

MODEL_PATH = 'my_model' # Assuming 'my_model' is in the same directory

def test_load_keras_model():
    print(f"[INFO] Attempting to load Keras model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Keras model path does not exist: {MODEL_PATH}")
        return False
    try:
        model = load_model(MODEL_PATH)
        print("[SUCCESS] Keras model loaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load Keras model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_init_mediapipe_holistic():
    print("[INFO] Attempting to initialize MediaPipe Holistic model...")
    try:
        with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
            print("[SUCCESS] MediaPipe Holistic model initialized successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize MediaPipe Holistic model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("[INFO] --- Starting Keras Model Load Test ---")
    keras_ok = test_load_keras_model()
    print("[INFO] --- Finished Keras Model Load Test ---")

    print("\n[INFO] --- Starting MediaPipe Holistic Init Test ---")
    mediapipe_ok = test_init_mediapipe_holistic()
    print("[INFO] --- Finished MediaPipe Holistic Init Test ---")

    if keras_ok and mediapipe_ok:
        print("\n[SUCCESS] Both critical components (Keras model load, MediaPipe init) seem OK.")
    else:
        print("\n[FAILURE] One or more critical components failed. See errors above.")
