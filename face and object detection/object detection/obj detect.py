import cv2
import numpy as np
import os

# Prompt the user for the IP address of the mobile device running IP Webcam
camera_ip = input("Enter the IP address of your mobile device running IP Webcam: ")

# Construct the default camera URL with port 8080
camera_url = f'http://172.15.10.190:8080/video'

# Load YOLO model files (download these files from the YOLO website)
# Replace this line with the full path to yolov3.cfg
yolo_config = r"C:\Users\User\Desktop\lokesh proj\object detection\yolov3.cfg"

yolo_weights = r"C:\Users\User\Desktop\lokesh proj\object detection\yolov3.weights"  # Replace with your YOLO weights file
yolo_classes = r"C:\Users\User\Desktop\lokesh proj\object detection\coco.names"  # Replace with your YOLO classes file

# Create a directory for saving snapshots
output_folder = "snapshots"
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model and classes
net = cv2.dnn.readNet(yolo_weights, yolo_config)
classes = []

with open(yolo_classes, "r") as f:
    classes = f.read().strip().split("\n")

# Create a VideoCapture object with the camera URL
print("Attempting to connect to the camera feed...")
cap = cv2.VideoCapture(camera_url)

if not cap.isOpened():
    print("Failed to connect to the camera feed.")
else:
    print("Successfully connected to the camera feed.")

frame_counter = 0  # Counter for saving frames

# ... (previous code)

# ... (previous code)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to receive frame from the camera.")
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the YOLO model
    net.setInput(blob)

    # Get detections
    detections = net.forward()

    for detection in detections:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Calculate coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Draw bounding box and label
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{classes[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save frames when objects are detected
            frame_counter += 1
            filename = os.path.join(output_folder, f"frame_{frame_counter}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved snapshot as {filename}")

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    