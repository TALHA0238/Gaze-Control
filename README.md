Experience Revolutionary Technology:
**Eye and Hand Tracking with Speech Recognition**

This project is an eye-tracking and hand-tracking application with integrated speech recognition. It provides a comprehensive user interaction system by combining these three functionalities. Here is a detailed explanation:

Eye Tracking
Face Detection and Landmark Prediction**: Uses `dlib` to detect faces and predict facial landmarks.
Eye Aspect Ratio (EAR)**: Computes EAR to detect blinks. If the EAR falls below a threshold for a certain number of consecutive frames, a blink is detected.
Mouse Control: Tracks eye movements to control the mouse cursor. The application calibrates the user's gaze to specific points on the screen for accurate tracking.
Calibration: Guides the user to look at specific points on the screen to record initial gaze positions.

Hand Tracking
Hand Detection: Utilizes `MediaPipe` to detect and track hand landmarks.
Mouse Clicks: Detects the index finger tip to perform mouse clicks. If two fingers are detected up, it triggers a click.

Speech Recognition
Voice Commands: Uses the `speech_recognition` library to recognize voice commands. Commands include actions like clicking, scrolling, and moving the mouse.
Threading: Runs speech recognition in a separate thread to avoid blocking the main loop.

Data Logging
CSV Logging: Logs eye center positions and mouse coordinates to a CSV file for further analysis.

User Interaction
Toggle Voice Commands: Allows toggling of voice commands using the Enter key.
Visual Feedback: Displays visual feedback for eye and hand tracking on the screen.

External Files
`shape_predictor_68_face_landmarks.dat`**: Required for dlib's facial landmarks predictor. Download it from dlib's model zoo and place it in the appropriate directory.
`eye_tracking_log.csv`**: Used to log eye center positions and mouse coordinates. Created automatically when the application runs.

Libraries Used
`opencv-python` (cv2)
`dlib`
`pyautogui`
`numpy`
`scipy`
`mediapipe`
`speech_recognition`

Setup and Usage
1. Install Dependencies:
   ```sh
   pip install opencv-python dlib pyautogui numpy scipy mediapipe SpeechRecognition
   ```
2. Clone the Repository:
   ```sh
   git clone https://github.com/your-username/your-repository-name.git
   cd your-repository-name
   ```
3. Download and Place the Shape Predictor Model.
4. Run the Application:
   ```sh
   python .venv/Scripts/eyetracking.py
   ```

This project captures video from the webcam, processes each frame to detect and track eyes and hands, and listens for voice commands to enhance user interaction.
