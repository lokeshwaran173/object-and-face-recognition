import cv2
import numpy as np

# Prompt the user for the IP address of the mobile device running IP Webcam
camera_ip = input("Enter the IP address of your mobile device running IP Webcam: ")

# Construct the default camera URL with port 8080
camera_url = f'http://172.15.10.190:8080/video'

# Load YOLO model files (download these files from the YOLO website)
yolo_config = r"C:\Users\User\Desktop\lokesh proj\object detection\yolov3.cfg"

yolo_weights = r"C:\Users\User\Desktop\lokesh proj\object detection\yolov3.weights"  # Replace with your YOLO weights file
yolo_classes = r"C:\Users\User\Desktop\lokesh proj\object detection\coco.names"  # Replace with your YOLO classes file
# Load YOLO model and classes
net = cv2.dnn.readNet(yolo_weights, yolo_config)
classes = []

with open(yolo_classes, "r") as f:
    classes = f.read().strip().split("\n")

# Create a VideoCapture object with the camera URL
cap = cv2.VideoCapture(camera_url)

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
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Calculate coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Draw bounding box and label
                color = (0, 255, 0)  # Green
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
