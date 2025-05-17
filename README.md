# Real-Time Sign Language Gesture Recognizer

This project captures video from a webcam, uses MediaPipe to detect hand landmarks, and a custom-trained Keras LSTM model to recognize a predefined set of sign language gestures in real-time. The recognized gestures are concatenated to form sentences, which can optionally be grammar-checked.

## Project Structure

- `main.py`: Main script to run the real-time sign gesture recognition.
- `data_collection.py`: Script to collect training data (landmark sequences) for new signs.
- `model.py`: Script to train the Keras LSTM model using the collected data. It saves the trained model as `my_model`.
- `my_functions.py`: Contains helper functions for MediaPipe processing and landmark extraction.
- `test_critical_components.py`: A utility script to test Keras model loading and MediaPipe initialization.
- `minimal_camera_mediapipe_test.py`: A utility script for basic camera and MediaPipe functionality testing.
- `data/`: Directory where collected landmark data (`.npy` files) for each sign/action is stored. This directory is created by `data_collection.py`.
- `my_model/`: Directory where the trained Keras model is saved by `model.py`.
- `sign_venv_v2/`: Python virtual environment directory (if created using the suggested setup).
- `requirements.txt`: Lists the Python dependencies.

## Setup and Usage

### 1. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment.

```bash
python3 -m venv sign_venv_v2
source sign_venv_v2/bin/activate 
# For Windows: .\sign_venv_v2\Scripts\activate
```

### 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```
*Note: On macOS with Apple Silicon (ARM), specific versions or installation methods (like using `tensorflow-macos`) might be necessary for `tensorflow` and `mediapipe` to work correctly due to native library dependencies. The `requirements.txt` aims for a generally compatible set.*

### 3. Data Collection

Before training a model, you need to collect data for the signs you want to recognize.
- Modify the `actions` list in `data_collection.py` to include the signs you want to record (e.g., `actions = np.array(["Hello", "Thanks", "Yes"])`).
- Run the data collection script:
  ```bash
  python data_collection.py
  ```
- The script will open your webcam. For each sign and each sequence (default 30 sequences per sign, 10 frames per sequence):
    - It will display: "Pause. Press 'Space' when you are ready."
    - Position yourself to make the sign clearly.
    - Press the **Spacebar**. The script will record 10 frames.
    - Repeat for all sequences and all signs.
- This will populate the `data/` directory with `.npy` files containing the landmark data.

### 4. Train the Model

Once data collection is complete, train the LSTM model:

```bash
python model.py
```
- This script will load the data from the `data/` directory, train the model, and save the trained model to a directory named `my_model`.

### 5. Run Real-Time Recognition

After the model is trained and `my_model` exists, run the main application:

```bash
python main.py
```
- This will open your webcam.
- Perform the signs you trained the model on.
- The recognized signs will be displayed and concatenated into a sentence.
- Press **Enter** to attempt grammar correction on the current sentence.
- Press **Spacebar** to clear the current sentence.

## Notes

- **Camera Permissions:** On macOS and some Linux distributions, you might need to grant camera access permissions to your terminal or IDE.
- **Lighting and Background:** Ensure good lighting and a non-cluttered background for better hand tracking and gesture recognition.
- **Sign Consistency:** Perform signs consistently during data collection for better model accuracy.
- **Customization:**
    - `actions` in `data_collection.py` and `model.py` (must match).
    - `sequences` and `frames` in `data_collection.py` and `model.py` (must match).
    - LSTM model architecture in `model.py`.
    - `min_detection_confidence` and `min_tracking_confidence` for MediaPipe in all scripts.
