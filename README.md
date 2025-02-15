# Smart Assist: YOLO-Based Object Detection for the Visually Impaired

## 📌 Project Overview
**Vision Partner** is an AI-powered assistive system that helps visually impaired individuals navigate their surroundings. It uses **YOLO (You Only Look Once) object detection** to recognize objects in real-time and provides **audio feedback** to the user. This project enhances independent mobility and accessibility through the integration of **computer vision and text-to-speech (TTS) technologies**.

## 🚀 Features
- 🎯 **Real-time object detection** using YOLOv4/YOLOv5  
- 🔊 **Audio feedback** via Text-to-Speech (TTS)  
- 🎥 **Live camera feed processing**  
- 📏 **Distance estimation** (optional for enhanced safety)  
- 🏗️ **Portable & lightweight model** for real-world deployment  

## 🛠️ Technologies Used
- **Deep Learning** – YOLO (You Only Look Once)  
- **Python** – OpenCV, TensorFlow/PyTorch  
- **Computer Vision** – Real-time image processing  
- **Text-to-Speech (TTS)** – gTTS or Google Cloud TTS  

## 📥 Installation & Setup
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Thiru-1820/Object_detection-using-Yolov10-.git
cd Object_detection-using-Yolov10
python object_detection.py (only for object detections)
python object_detection_distance.py (object detection with distancd)
python object_detection_distance_tts.py 
