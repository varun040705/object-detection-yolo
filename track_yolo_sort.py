import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('C:/Users/KML/sort')

from sort import Sort
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8n.pt')
tracker = Sort()

# Try camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Camera not found. Using fallback video.")
    cap = cv2.VideoCapture("C:/Users/KML/Desktop/sample_video.mp4")

if not cap.isOpened():
    raise Exception("❌ Could not open camera or fallback video.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or cannot read frame.")
        break

    results = model(frame, verbose=False)[0]

    detections = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            detections.append([x1.item(), y1.item(), x2.item(), y2.item(), conf.item()])

    if len(detections) > 0:
        tracked_objects = tracker.update(np.array(detections))
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, f'ID {int(obj_id)}', (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow('YOLOv8 + SORT Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
