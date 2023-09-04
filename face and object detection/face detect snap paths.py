import cv2
import os

# Prompt the user for the IP address of the mobile device running IP Webcam
camera_ip = input("Enter the IP address of your mobile device running IP Webcam: ")

# Construct the default camera URL with port 8080
camera_url = f'http://172.15.10.190:8080/video'

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create a VideoCapture object with the camera URL
cap = cv2.VideoCapture(camera_url)

# Initialize a counter for captured faces
face_count = 0

# Create a directory named 'snapshots' if it doesn't exist
if not os.path.exists('snapshots'):
    os.makedirs('snapshots')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame from the camera.")
        break


    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces and save them as images
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Crop the detected face region
        detected_face = frame[y:y+h, x:x+w]

        # Save the detected face as an image in the 'snapshots' folder
        face_filename = os.path.join('snapshots', f"face_{face_count}.jpg")
        cv2.imwrite(face_filename, detected_face)
        print(f"Saved {face_filename}")

        # Increment the face count
        face_count += 1

    # Display the frame
    cv2.imshow('Mobile Device Camera with Face Detection', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
