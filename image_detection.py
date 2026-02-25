from ultralytics import YOLO
import cv2

model=YOLO('yolov8n.pt')
results=model(r"C:\Users\KML\Desktop\intern works\bus.jpg")

result=results[0].plot()
cv2.imshow("detection",result)
cv2.waitKey(0)