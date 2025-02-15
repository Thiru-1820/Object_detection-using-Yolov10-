import cv2
from ultralytics import YOLO
import numpy as np
import pyttsx3



# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

# Load YOLO model
model = YOLO('yolov10n.pt')

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

focal_length = 600  

object_widths = {
    'cell phone': 8.0,   
    'bottle': 6.0,  
    'id card': 8.5,  
    'person': 45.0 
}


def get_distance(bbox_width_pixels, real_object_width_cm):
    if bbox_width_pixels > 0:  # Avoid division by zero
        distance_cm = (real_object_width_cm * focal_length) / bbox_width_pixels
        return distance_cm
    return None


last_detected_label = None
last_announcement_time = 0
announcement_interval = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    
    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  
        scores = result.boxes.conf.cpu().numpy()  
        classes = result.boxes.cls.cpu().numpy().astype(int)  
        labels = result.names  

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            bbox_width = x2 - x1  

            
            current_label = labels[cls]

            
            if current_label in object_widths:
            
                real_object_width_cm = object_widths[current_label]

            
                distance = get_distance(bbox_width, real_object_width_cm)

                
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                frame = cv2.putText(frame, current_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                
                if distance is not None:
                    distance_text = f"Distance: {distance:.2f} cm"
                    frame = cv2.putText(frame, distance_text, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                
                    if current_label != last_detected_label or (cv2.getTickCount() / cv2.getTickFrequency()) - last_announcement_time > announcement_interval:
                        tts_engine.say(f"{current_label} detected at a distance of {distance:.2f} centimeters.")
                        tts_engine.runAndWait()
                        last_detected_label = current_label
                        last_announcement_time = cv2.getTickCount() / cv2.getTickFrequency()

    # Display the resulting frame
    cv2.imshow('YOLO Webcam Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
