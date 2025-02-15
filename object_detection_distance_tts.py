import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3
import time

# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Load YOLO model
model = YOLO(r'D:\sem5\project\implementation\git\weights\yolov10n.pt')

# Open webcam with DirectShow backend to avoid MSMF errors
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define constants for distance calculation
focal_length = 600  # Adjust this based on your camera
distance_threshold = 50  # Minimum pixel distance to consider a new object

# Mapping of object labels to their real-world widths in cm
object_widths = {
    'cell phone': 8.0,   # Width of a phone in cm
    'bottle': 6.0,  # Width of a bottle in cm
    'id card': 8.5,  # Width of an ID card in cm
    'person': 45.0  # Approximate width of a person (shoulder width) in cm
}

# Dictionary to store unique labels for each detected object
object_counter = {}

# List to store previously detected bounding boxes and labels
detected_objects = {}

# Function to calculate the distance
def get_distance(bbox_width_pixels, real_object_width_cm):
    if bbox_width_pixels > 0:  # Avoid division by zero
        distance_cm = (real_object_width_cm * focal_length) / bbox_width_pixels
        return distance_cm
    return None

# Function to create a unique label for a new object
def get_unique_label(base_label):
    if base_label not in object_counter:
        object_counter[base_label] = 0
    object_counter[base_label] += 1
    return f"{base_label}{object_counter[base_label]}"

# Function to check if two boxes are far enough to be considered different objects
def is_new_object(new_box, existing_boxes, threshold=distance_threshold):
    for box in existing_boxes:
        distance = np.linalg.norm(np.array(new_box[:2]) - np.array(box[:2]))
        if distance < threshold:
            return False
    return True

# Variable to track the last detected object and time for announcing
last_detected_label = None
last_announcement_time = 0
announcement_interval = 5  # Time in seconds between announcements

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Run YOLO object detection
    results = model(frame)

    current_boxes = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Get confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Get class indices
        labels = result.names  # Labels (names of the classes)

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1  # Calculate the bounding box width in pixels
            current_label = labels[cls]  # Get the current label for display

            # Only process specified object labels for distance calculation
            if current_label in object_widths:
                real_object_width_cm = object_widths[current_label]
                distance = get_distance(bbox_width, real_object_width_cm)

                # Check if this is a new object or an existing one
                if is_new_object([x1, y1, x2, y2], current_boxes):
                    unique_label = get_unique_label(current_label)  # Generate unique label for new objects
                    current_boxes.append([x1, y1, x2, y2])  # Add box to current boxes

                    # Add the object to the list of detected objects with its unique label
                    detected_objects[unique_label] = [x1, y1, x2, y2]

                    # Draw bounding box and unique label
                    frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    frame = cv2.putText(frame, unique_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # Display distance if available
                    if distance is not None:
                        distance_text = f"Distance: {distance:.2f} cm"
                        frame = cv2.putText(frame, distance_text, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Announce the detected object if it's new or a significant time has passed
                        current_time = time.time()
                        if unique_label != last_detected_label or (current_time - last_announcement_time) > announcement_interval:
                            tts_engine.say(f"{unique_label} detected at a distance of {distance:.2f} centimeters.")
                            tts_engine.runAndWait()
                            last_detected_label = unique_label
                            last_announcement_time = current_time

    # Display the resulting frame
    cv2.imshow('YOLO Webcam Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
