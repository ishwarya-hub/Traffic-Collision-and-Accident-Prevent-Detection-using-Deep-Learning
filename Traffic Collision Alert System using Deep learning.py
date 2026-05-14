import cv2
import numpy as np
from ultralytics import YOLO
import math

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use yolov8n.pt or yolov8s.pt

# Classes: 0 = person, 2-7 = vehicles (car, motorcycle, bus, truck)
VEHICLE_CLASSES = [2, 3, 5, 7]
PERSON_CLASS = 0
DISTANCE_THRESHOLD = 100  # Pixel distance for collision warning

# Video capture
cap = cv2.VideoCapture("input_video.mp4")  # Replace with 0 for webcam

def calculate_distance(box1, box2):
    x1, y1 = int((box1[0] + box1[2]) / 2), int((box1[1] + box1[3]) / 2)
    x2, y2 = int((box2[0] + box2[2]) / 2), int((box2[1] + box2[3]) / 2)
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, stream=True)
    
    vehicles = []
    people = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if cls in VEHICLE_CLASSES:
                vehicles.append([x1, y1, x2, y2])
            elif cls == PERSON_CLASS:
                people.append([x1, y1, x2, y2])

    # Collision detection
    warning = False
    for v_box in vehicles:
        for p_box in people:
            distance = calculate_distance(v_box, p_box)
            if distance < DISTANCE_THRESHOLD:
                warning = True
                cv2.putText(frame, "⚠️ Collision Warning!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(frame, (v_box[0], v_box[1]), (v_box[2], v_box[3]), (0, 0, 255), 2)
                cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 0, 255), 2)

    # Draw remaining boxes
    for box in vehicles:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    for box in people:
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    cv2.imshow("Traffic Collision Detection", frame)

    # Exit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
