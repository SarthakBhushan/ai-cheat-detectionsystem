# 🎓 AI Cheat Detection System

An intelligent exam monitoring system that uses **OpenCV** and **YOLO** to detect cheating behavior in real-time via webcam or video feed. The system identifies suspicious activities such as abnormal eye/head movement, phone usage, and unusual body posture during examinations.

---

## 🚀 Features

- 👁️ **Gaze & Pupil Tracking** — Detects when a student's eyes deviate significantly from the screen
- 📱 **Phone Detection** — Identifies mobile phone usage during exams using YOLOv8/YOLO11
- 🧍 **Pose Estimation** — Monitors body posture for suspicious movement using YOLO pose models
- 🌐 **Web Dashboard** — HTML-based frontend to view alerts and monitoring status
- 🖥️ **Flask Server** — Backend server to process video streams and serve detection results

---

## 🗂️ Project Structure

```
ai-cheat-detectionsystem/
│
├── final1.py                  # Main detection pipeline (combined modules)
├── phone-detector.py          # Phone detection using YOLO object detection
├── pupil5.py                  # Pupil/gaze tracking using OpenCV
├── server (4) (1) (1).py      # Flask server for handling video stream & API
├── hack-n-forge (6).html      # Frontend web dashboard
├── requirements.txt           # Python dependencies
│
├── yolo11n.pt                 # YOLO11 Nano object detection model
├── yolo11n-pose.pt            # YOLO11 Nano pose estimation model
└── yolov8n-pose.pt            # YOLOv8 Nano pose estimation model
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Computer Vision | OpenCV |
| Object Detection | YOLOv8 / YOLO11 (Ultralytics) |
| Pose Estimation | YOLOv8-Pose / YOLO11-Pose |
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip
- Webcam or video input device

### Steps

1. **Clone the repository**

   ```bash
   git clone https://github.com/SarthakBhushan/ai-cheat-detectionsystem.git
   cd ai-cheat-detectionsystem
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**

   ```bash
   python "server (4) (1) (1).py"
   ```

4. **Open the dashboard**

   Open `hack-n-forge (6).html` in your browser, or navigate to the local server URL shown in the terminal.

---

## 🔍 Detection Modules

### 👁️ Pupil & Gaze Tracking (`pupil5.py`)
Uses OpenCV's eye detection to track pupil position. Raises a flag when the student's gaze moves away from the screen for a sustained period, indicating possible copying behavior.

### 📱 Phone Detection (`phone-detector.py`)
Leverages `yolo11n.pt` to detect mobile phones in the camera frame in real time. Triggers an alert when a phone is visible.

### 🧍 Pose Estimation (`final1.py`)
Uses `yolo11n-pose.pt` or `yolov8n-pose.pt` to analyze body keypoints. Detects suspicious postures such as turning the head/body sideways, which may indicate communication with another person.

### 🌐 Server (`server.py`)
A Flask-based backend that:
- Streams video from the webcam
- Runs all detection modules
- Serves alert data to the frontend dashboard

---

## 📋 Requirements

Key dependencies (from `requirements.txt`):

```
ultralytics
opencv-python
flask
numpy
```

---

## 🖼️ How It Works

```
Webcam Feed
    │
    ▼
OpenCV Frame Capture
    │
    ├──► Pupil/Gaze Tracker ──► Suspicious Gaze? ──► Alert
    │
    ├──► YOLO Phone Detector ──► Phone Visible? ──► Alert
    │
    └──► YOLO Pose Estimator ──► Suspicious Pose? ──► Alert
                │
                ▼
         Flask Server (API)
                │
                ▼
         Web Dashboard (HTML)
```

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes** only. Any deployment in real examination environments should comply with local privacy laws and institutional policies regarding student monitoring.

---

## 🙌 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project currently has no license specified. Please contact the repository owner before using it in production.

---

*Built with ❤️ using OpenCV and YOLO*
