# 🚀 Real-Time Object Detection Web App using YOLOv8

## 📌 Overview
This project implements a real-time object detection system using YOLOv8, capable of detecting multiple objects in video streams with high speed and accuracy.

An interactive web application is built using Streamlit, allowing users to upload videos, process them frame-by-frame, and download the annotated output.

---

## 🛠 Tech Stack
- Python  
- OpenCV  
- Streamlit  
- YOLOv8 (Ultralytics)

---

## ⚙️ Features
- 📹 Upload video files (mp4, avi, mov, mkv)  
- 🎯 Real-time object detection using YOLOv8  
- 📊 Live progress bar with FPS and ETA tracking  
- 🧠 Detection of multiple objects with confidence scores  
- 💾 Download processed video output  
- 🖥️ Simple and interactive UI  

---

## 🧠 How It Works
- Video is read frame-by-frame using OpenCV  
- Each frame is passed into the YOLOv8 model  
- Objects are detected and annotated with bounding boxes  
- Processed frames are written into a new video file  
- Output video is displayed and available for download  

---

## 📸 Results
(Add your output screenshots here)

Example detections include:
- Person  
- Bicycle  
- Dog  
- Traffic light  
- Handbag  

---

## 🚀 Applications
- Smart surveillance systems  
- Drone vision and navigation  
- Autonomous systems  
- Traffic and crowd monitoring  

YOLO is widely used in real-time applications because it can detect objects in a single forward pass, making it fast and efficient for video processing.

---

## 🔧 Installation

```bash
pip install ultralytics streamlit opencv-python

streamlit run app.py
