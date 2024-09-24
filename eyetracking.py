import cv2
import dlib
import pyautogui
import numpy as np
from scipy.spatial import distance as dist
from collections import deque
import csv
import mediapipe as mp
import speech_recognition as sr
import threading

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    "C:/Users/Dell/PycharmProjects/Eye Tracking/.venv/Scripts/shape_predictor_68_face_landmarks.dat")  # Update with the correct path to your .dat file

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to compute Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for blink detection
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 3
blink_counter = 0
total_blinks = 0
click_triggered = False

# Indices for eyes landmarks in the 68-point facial landmark model
LEFT_EYE_INDICES = list(range(36, 42))
RIGHT_EYE_INDICES = list(range(42, 48))

# Capture webcam video
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Calibration step: Use this to record initial gaze positions
calibration_points = []  # List to store calibration points

def calibrate():
    """Guide the user to look at specific points on the screen for calibration."""
    screen_width, screen_height = pyautogui.size()

    # Display instructions for the user to look at different screen positions
    calibration_instructions = [
        ("Top Left", (0.1 * screen_width, 0.1 * screen_height)),
        ("Top Right", (0.9 * screen_width, 0.1 * screen_height)),
        ("Bottom Left", (0.1 * screen_width, 0.9 * screen_height)),
        ("Bottom Right", (0.9 * screen_width, 0.9 * screen_height)),
        ("Center", (0.5 * screen_width, 0.5 * screen_height)),
    ]

    for instruction, pos in calibration_instructions:
        print(f"Look at {instruction} of the screen for 3 seconds")
        pyautogui.moveTo(pos[0], pos[1])
        cv2.waitKey(3000)  # Wait 3 seconds per point
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES]
            calibration_points.append((left_eye, right_eye))
        else:
            print("No face detected. Please recalibrate.")
            break

calibrate()

# Deques to store the last few eye center positions for smoothing
left_eye_centers = deque(maxlen=5)
right_eye_centers = deque(maxlen=5)

# Open a CSV file to log data
log_file = open('eye_tracking_log.csv', 'w', newline='')
log_writer = csv.writer(log_file)
log_writer.writerow(['Timestamp', 'Left Eye Center', 'Right Eye Center', 'Mouse X', 'Mouse Y'])

# List to store gaze points for heatmap
gaze_points = []

# Initialize the recognizer
recognizer = sr.Recognizer()

def recognize_speech():
    """Recognize speech and perform actions based on commands."""
    while True:
        with sr.Microphone() as source:
            print("Listening for commands...")
            audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")

            if "click" in command:
                pyautogui.click()
            elif "scroll up" in command:
                pyautogui.scroll(100)
            elif "scroll down" in command:
                pyautogui.scroll(-100)
            elif "left" in command:
                pyautogui.moveRel(-50, 0)
            elif "right" in command:
                pyautogui.moveRel(50, 0)
            elif "up" in command:
                pyautogui.moveRel(0, -50)
            elif "down" in command:
                pyautogui.moveRel(0, 50)
            else:
                print("Command not recognized.")
        except sr.UnknownValueError:
            print("Could not understand the command.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

# Start the speech recognition in a separate thread
speech_thread = threading.Thread(target=recognize_speech)
speech_thread.daemon = True
speech_thread.start()

# Process each frame for tracking
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Get coordinates for the left and right eye
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE_INDICES]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE_INDICES]

        # Calculate EAR for blink detection
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)

        # Average EAR of both eyes
        avg_ear = (left_ear + right_ear) / 2.0

        # Detect blink (if EAR is below threshold for consecutive frames)
        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                click_triggered = True
                total_blinks += 1
                print(f"Blink detected, triggering click. Total blinks: {total_blinks}")
                pyautogui.click()
            blink_counter = 0

        # Reset click_triggered flag
        if click_triggered:
            click_triggered = False

        # Draw eye landmarks
        for pt in left_eye + right_eye:
            cv2.circle(frame, pt, 2, (0, 255, 0), -1)

        # Use eye movement to control mouse
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)

        # Append the current eye centers to the deques
        left_eye_centers.append(left_eye_center)
        right_eye_centers.append(right_eye_center)

        # Calculate the average of the last few eye centers for smoothing
        smoothed_left_eye_center = np.mean(left_eye_centers, axis=0)
        smoothed_right_eye_center = np.mean(right_eye_centers, axis=0)

        # Move mouse according to smoothed eye movement and draw visual feedback
        screen_width, screen_height = pyautogui.size()
        mouse_x = int(smoothed_left_eye_center[0] * screen_width / frame.shape[1])
        mouse_y = int(smoothed_left_eye_center[1] * screen_height / frame.shape[0])
        pyautogui.moveTo(mouse_x, mouse_y)
        cv2.circle(frame, (mouse_x, mouse_y), 10, (255, 0, 0), -1)  # Draw circle at mouse position

        # Append gaze points
        gaze_points.append((mouse_x, mouse_y))

        # Draw heatmap
        for point in gaze_points:
            cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Red dots for heatmap

        # Log data
        log_writer.writerow([cv2.getTickCount(), smoothed_left_eye_center, smoothed_right_eye_center, mouse_x, mouse_y])

    # Convert the frame to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            index_finger_tip_x = int(index_finger_tip.x * w)
            index_finger_tip_y = int(index_finger_tip.y * h)

            # Perform click if index finger is shown
            pyautogui.click()
            print("Index finger detected, performing click")

            # Draw hand landmarks
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow('Eye Tracking', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        dynamic_calibrate()

cap.release()
cv2.destroyAllWindows()
log_file.close()