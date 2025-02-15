import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLO model
model = YOLO('/yolov10n.pt')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define constants for distance calculation
focal_length =850  # Adjust this based on your camera
real_object_width_cm = 8.5  # Width of the actual object in cm

# Function to calculate the distance
def get_distance(bbox_width_pixels):
    distance_cm = (real_object_width_cm * focal_length) / bbox_width_pixels
    return distance_cm

# Variable to track the last detected object
last_detected_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO object detection
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Get class indices
        labels = result.names  # Labels (names of the classes)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1  # Calculate the bounding box width in pixels

            # Calculate distance
            distance = get_distance(bbox_width)

            # Get current label for display
            current_label = f'{labels[cls]} {score:.2f}'

            # Draw bounding box and label
            frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            frame = cv2.putText(frame, current_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Update distance display only if a new object is detected
            if current_label != last_detected_label:
                last_detected_label = current_label  # Update last detected label
                distance_text = f"Distance: {distance:.2f} cm"
                # Display distance on the image
                frame = cv2.putText(frame, distance_text, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('YOLO Webcam Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()