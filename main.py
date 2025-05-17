# %%

# Import necessary libraries
import numpy as np
import os
import string
import mediapipe as mp
import cv2
from my_functions import *
import sys
from tensorflow.keras.models import load_model
import language_tool_python
import sys

# Set the path to the data directory
PATH = os.path.join('data')

# Create an array of action labels by listing the contents of the data directory
actions = np.array(os.listdir(PATH))

# Load the trained model
MODEL_PATH = 'my_model'
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model directory not found: '{MODEL_PATH}'.")
    print("Please train the model first by running: python model.py")
    sys.exit(1)
model = load_model(MODEL_PATH)

# Create an instance of the grammar correction tool
tool = language_tool_python.LanguageToolPublicAPI('en-UK')

# Initialize the lists
sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a holistic object for sign prediction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Run the loop while the camera is open
    while cap.isOpened():
        # Read a frame from the camera
        ret, image = cap.read()
        # If frame not read correctly, exit the loop
        if not ret or image is None:
            print("[ERROR] Failed to grab frame from camera. Exiting.")
            break
        # Process the image and obtain sign landmarks using image_process function from my_functions.py
        results = image_process(image, holistic)
        # Draw the sign landmarks on the image using draw_landmarks function from my_functions.py
        draw_landmarks(image, results)
        # Extract keypoints from the pose landmarks using keypoint_extraction function from my_functions.py
        keypoints.append(keypoint_extraction(results))

        # Check if 10 frames have been accumulated
        if len(keypoints) == 10:
            # Convert keypoints list to a numpy array
            keypoints = np.array(keypoints)
            # Make a prediction on the keypoints using the loaded model
            prediction = model.predict(keypoints[np.newaxis, :, :])
            # Clear the keypoints list for the next set of frames
            keypoints = []

            # Check if the maximum prediction probability is above threshold
            if np.amax(prediction) > 0.9:
                pred_index = np.argmax(prediction)
                pred_action = actions[pred_index]
                # Only record if different from last prediction
                if last_prediction != pred_action:
                    sentence.append(pred_action)
                    last_prediction = pred_action
                    # Print prediction and current sentence to console
                    print(f"[PREDICT] {pred_action}")
                    print(f"[SENTENCE] {' '.join(sentence)}")

        # Limit the sentence length to 7 elements to make sure it fits on the screen
        if len(sentence) > 7:
            sentence = sentence[-7:]

        # Wait for a key press for 10ms
        key = cv2.waitKey(10) & 0xFF
        # Reset if the "Spacebar" is pressed
        if key == ord(' '):
            sentence, keypoints, last_prediction, grammar, grammar_result = [], [], [], [], []

        # Check if the list is not empty
        if sentence:
            # Capitalize the first word of the sentence
            sentence[0] = sentence[0].capitalize()

        # Check if the sentence has at least two elements
        if len(sentence) >= 2:
            # Check if the last element of the sentence belongs to the alphabet (lower or upper cases)
            if sentence[-1] in string.ascii_lowercase or sentence[-1] in string.ascii_uppercase:
                # Check if the second last element of sentence belongs to the alphabet or is a new word
                if sentence[-2] in string.ascii_lowercase or sentence[-2] in string.ascii_uppercase or (sentence[-2] not in actions and sentence[-2] not in list(x.capitalize() for x in actions)):
                    # Combine last two elements
                    sentence[-1] = sentence[-2] + sentence[-1]
                    sentence.pop(len(sentence) - 2)
                    sentence[-1] = sentence[-1].capitalize()

        # Perform grammar check if "Enter" is pressed
        elif key in (10, 13):  # Enter key codes
            text = ' '.join(sentence)
            grammar_result = tool.correct(text)
            # Print corrected grammar to console
            print(f"[GRAMMAR] {grammar_result}")

        if grammar_result:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(grammar_result, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, grammar_result, (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            # Calculate the size of the text to be displayed and the X coordinate for centering the text on the image
            textsize = cv2.getTextSize(' '.join(sentence), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_X_coord = (image.shape[1] - textsize[0]) // 2

            # Draw the sentence on the image
            cv2.putText(image, ' '.join(sentence), (text_X_coord, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show the image on the display
        cv2.imshow('Camera', image)

        # Check if the 'Camera' window was closed or Esc key pressed to break the loop
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1 or key == 27:
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()

    # Shut off the server
    tool.close()
