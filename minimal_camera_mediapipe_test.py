import cv2
import mediapipe as mp
import time

print("[INFO] Attempting to initialize MediaPipe Holistic...")
try:
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75)
    print("[SUCCESS] MediaPipe Holistic initialized.")
except Exception as e:
    print(f"[ERROR] Failed to initialize MediaPipe Holistic: {e}")
    exit()

print("[INFO] Attempting to access camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access camera.")
    exit()
else:
    print("[SUCCESS] Camera accessed.")

print("[INFO] Will display camera feed for 5 seconds...")
start_time = time.time()
while cap.isOpened() and (time.time() - start_time) < 5: # Display for 5 seconds
    ret, frame = cap.read()
    if not ret:
        print("[WARNING] Failed to grab frame.")
        break
    
    # Minimal processing just to ensure pipeline works
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb) # Process with MediaPipe
    
    # Draw landmarks if any are detected (optional, but good test)
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    cv2.imshow('Minimal Test', frame)
    if cv2.waitKey(5) & 0xFF == 27: # Esc key to exit
        break

print("[INFO] Releasing camera and destroying windows.")
cap.release()
cv2.destroyAllWindows()
print("[INFO] Test finished.")
