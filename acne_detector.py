import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect acne in a frame
def detect_acne(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds of the acne color in HSV
    lower_acne = np.array([0, 100, 100])
    upper_acne = np.array([10, 255, 255])

    # Threshold the frame to get only acne color
    acne_mask = cv2.inRange(hsv, lower_acne, upper_acne)

    # Find contours in the acne mask
    contours, _ = cv2.findContours(acne_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around acne
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

while True:
    ret, frame = cap.read()

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Detect acne within the face region and draw rectangles
        face_with_acne_rectangles = detect_acne(face_roi)

        # Draw rectangles around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with face and acne detection using rectangles
    cv2.imshow('Face and Acne Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
