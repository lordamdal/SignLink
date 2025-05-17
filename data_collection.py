# %%

# Import necessary libraries
import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import *
# import keyboard # No longer using keyboard library

# Define the actions (signs) that will be recorded and stored in the dataset
actions = np.array(["Love", "Hello", "Goodbye", "ThankYou"])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence, frame in product(actions, range(sequences), range(frames)):
        # If it is the first frame of a sequence, wait for the spacebar key press to start recording
        if frame == 0:
            # Display "Press Space to Start" message and wait
            while True:
                ret, image = cap.read() # Check the return value 'ret'
                if not ret or image is None: # Proper check for image read
                    print("Error: Could not read frame from camera for pause screen.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                
                results = image_process(image, holistic)
                # Ensure image is writeable before drawing on it
                image.flags.writeable = True
                draw_landmarks(image, results)

                cv2.putText(image, 'Recroding data for the "{}". Sequence number {}.'.format(action, sequence),
                            (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Pause.', (20,400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Press "Space" when you are ready.', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('Camera', image)
                
                key_pressed = cv2.waitKey(100) & 0xFF # Increased wait time for responsiveness
                if key_pressed == ord(' '): # Spacebar
                    break
                elif key_pressed == 27: # ESC key
                    print("ESC pressed, exiting data collection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                
                # Check if the 'Camera' window was closed and break the loop
                if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
                    print("Camera window closed, exiting data collection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
        
        # For all frames (including the first one after space is pressed)
        ret, image = cap.read() # Check the return value 'ret'
        if not ret or image is None: # Proper check for image read
            print(f"Error: Could not read frame from camera during recording for {action}, seq {sequence}, frame {frame}.")
            continue # Skip this frame or handle error appropriately

        # Process the image and extract hand landmarks using the MediaPipe Holistic pipeline
        results = image_process(image, holistic)
        # Ensure image is writeable before drawing on it
        image.flags.writeable = True
        # Draw the hand landmarks on the image
        draw_landmarks(image, results)

        # Display text on the image indicating the action and sequence number being recorded
        cv2.putText(image, 'RECORDING data for "{}". Sequence {}. Frame {}.'.format(action, sequence, frame),
                    (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.imshow('Camera', image)
        
        # Allow a very short delay to ensure frame is displayed and processed
        # Crucially, waitKey is needed for imshow to update.
        # If ESC is pressed during recording, exit.
        if cv2.waitKey(30) & 0xFF == 27: # 30ms delay, also allows ESC to quit
            print("ESC pressed during recording, exiting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Check if the 'Camera' window was closed and break the loop
        if cv2.getWindowProperty('Camera',cv2.WND_PROP_VISIBLE) < 1:
            print("Camera window closed during recording, exiting data collection.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

        # Extract the landmarks from both hands and save them in arrays
        keypoints = keypoint_extraction(results)
        frame_path = os.path.join(PATH, action, str(sequence), str(frame))
        try:
            np.save(frame_path, keypoints)
            print(f"Saved: {frame_path}.npy") # Added print statement
        except Exception as e:
            print(f"Error saving {frame_path}.npy: {e}") # Added error print

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
