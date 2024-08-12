import cv2
import numpy as np
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# Constants
EYE_AR_THRESH = 0.9
EYE_AR_CONSEC_FRAMES = 48
LED_PIN = 27  # GPIO pin for the LED

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)  # Start with the LED off

# Initialize variables
COUNTER = 0
ALARM_ON = False

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define indexes for the left and right eye from the MediaPipe landmarks
LEFT_EYE_IDX = [33, 133, 160, 158, 144, 153]  # Left eye landmarks
RIGHT_EYE_IDX = [362, 263, 387, 385, 373, 380]  # Right eye landmarks

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    
    return ear

# Initialize the camera
cap = cv2.VideoCapture(0)

try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face landmarks using MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract the left and right eye coordinates
                leftEye = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) for idx in LEFT_EYE_IDX])
                rightEye = np.array([(face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y) for idx in RIGHT_EYE_IDX])
                
                # Convert normalized coordinates to pixel coordinates
                leftEye *= np.array([frame.shape[1], frame.shape[0]])
                rightEye *= np.array([frame.shape[1], frame.shape[0]])
                
                # Compute the eye aspect ratio for both eyes
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                
                # Average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                
                # Draw the convex hull around the eyes
                cv2.polylines(frame, [leftEye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.polylines(frame, [rightEye.astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=1)
                
                # Check if the eye aspect ratio is below the blink threshold
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    
                    # If the eyes have been closed for a sufficient number of frames
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not ALARM_ON:
                            ALARM_ON = True
                            GPIO.output(LED_PIN, GPIO.HIGH)  # Turn on the LED
                            print("Drowsiness Detected!")
                            
                        # Draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    COUNTER = 0
                    if ALARM_ON:
                        ALARM_ON = False
                        GPIO.output(LED_PIN, GPIO.LOW)  # Turn off the LED
                
                # Display the computed eye aspect ratio
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Frame", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()  # Reset GPIO settings
