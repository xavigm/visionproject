# 🛰️ Real-Time Computer Vision & Detection Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8%2F12.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![Supervision](https://img.shields.io/badge/Roboflow-Supervision-purple?style=for-the-badge)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX_3060_12GB-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

![Header](visionproject.jpg)

An end-to-end, high-performance real-time object detection, classification, and analytics system powered by **RF-DETR (Roboflow Detection Transformer)** and **Roboflow Supervision**. Optimized for local GPU acceleration (NVIDIA RTX 3060 12GB), this application provides a modern web interface with real-time video streaming (MJPEG), live category aggregation (People, Vehicles, Animals), and temporal analytics using SQLite and Chart.js.

---

## 📸 Overview & Features

- **🚀 GPU-Accelerated Inference**: Leverages CUDA and PyTorch for high FPS processing of images, video files, USB webcams, and IP/RTSP online streams.
- **🎯 Precise Multi-Class Tracking**: Resolves COCO class mapping issues by filtering predictions via string class labels (`person`, `car`, `bus`, `dog`, etc.).
- **📊 Real-Time HTML5 Dashboard**: Built with Bootstrap 5 and Chart.js to render live detections and hourly historical trends.
- **🗄️ Automated Metrics Logging**: Background database thread writes aggregated detection metrics to an SQLite storage engine every 5 seconds.
- **⚡ MJPEG Video Streaming**: Low-latency video pipeline serving annotated frames over HTTP multipart streams.

---

## 🛠️ Architecture & Workflow

```text
               +-------------------------------------------------+
               |                Video/Image Input                |
               | (File Path / USB Webcam '0' / RTSP Live Stream) |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |            RF-DETR Inference Engine             |
               |             (CUDA / RTX 3060 12GB)              |
               +-------------------------------------------------+
                                       |
                                       v
               +-------------------------------------------------+
               |         Roboflow Supervision Annotator          |
               |      (BoxAnnotator + LabelAnnotator + Data)     |
               +-------------------------------------------------+
                                  /         \
                                 /           \
                                v             v
        +-------------------------------+   +-------------------------------+
        |  MJPEG Generator Feed (/video) |   | SQLite Logging Thread (5 sec) |
        +-------------------------------+   +-------------------------------+
                        |                                   |
                        v                                   v
        +-------------------------------+   +-------------------------------+
        |  HTML5 Video Player (Stream)   |   |   Chart.js Analytics Engine   |
        +-------------------------------+   +-------------------------------+
```

---

## 📋 Prerequisites & Installation

### 1. Hardware & System Requirements
- **OS**: Windows 10/11 or Ubuntu 20.04/22.04 LTS
- **GPU**: NVIDIA GPU (RTX 3060 12GB recommended or higher) with CUDA drivers installed.
- **Python**: Python 3.10+

### 2. Clone Repository & Environment Setup
```bash
git clone https://github.com/your-username/rfdetr-vision-dashboard.git
cd rfdetr-vision-dashboard

# Create virtual environment
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate
# Activate on Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Web, Vision, and Analytics Libraries
pip install flask opencv-python supervision rfdetr numpy
```

---

## 📂 Project Structure

```text
rfdetr-vision-dashboard/
├── app.py                  # Flask Application, GPU pipeline, and SQLite API
├── database.db             # SQLite database (Auto-generated on first run)
├── static/
│   └── js/
│       └── dashboard.js    # Chart.js initialization and polling script
├── templates/
│   └── index.html          # Responsive Bootstrap 5 Dashboard UI
└── README.md               # Project documentation
```

---

## 🚀 Usage Guide

### 1. Launch the Application
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

### 2. Stream Inputs
In the input box at the top of the dashboard, enter any of the following stream formats and click **"Iniciar Stream"**:

| Input Type | Input String Example | Description |
| :--- | :--- | :--- |
| **Local Image** | `ciudad.jpg` or `C:/path/to/image.png` | Single frame static detection |
| **Local Video** | `traffic.mp4` or `/var/media/demo.mkv` | Loopable video file stream |
| **USB Webcam** | `0` or `1` | Real-time camera device index |
| **IP / RTSP Stream** | `rtsp://admin:pass@192.168.1.50:554/stream1` | Security camera or online stream |
| **HTTP Live Stream** | `http://192.168.1.100:8080/video` | IP Webcam app stream |

---

## ⚙️ Fine-Tuning & Model Optimization

### 1. Model Sizes & Trade-Offs
In `app.py`, you can change the model class depending on your speed and accuracy requirements:

```python
from rfdetr import RFDETRBase, RFDETRSmall, RFDETRMedium, RFDETRLarge

# Select your model according to your GPU capacity:
model = RFDETRSmall()   # Fast & lightweight (~10-15ms / frame on RTX 3060)
# model = RFDETRMedium()  # Balanced accuracy and speed
# model = RFDETRLarge()   # High precision for small/distant objects (requires higher VRAM)
```

### 2. Tuning Confidence Thresholds
If the model misses small objects (under-detection) or generates false positives:
- **Decrease threshold** (e.g., `0.20` - `0.25`) to catch smaller or occluded objects/people in distant street views.
- **Increase threshold** (e.g., `0.45` - `0.60`) to reduce false positives in crowded, noisy environments.

In `app.py`:
```python
# Adjust conf_threshold during prediction
detections = model.predict(rgb_frame, conf_threshold=0.25)
```

### 3. Custom Class Grouping
You can modify or extend category groupings by editing the class name sets in `app.py`:

```python
PERSON_CLASSES = {"person"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}
ANIMAL_CLASSES = {"dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear"}
```

---

## 📊 Database & API Endpoints

- **`GET /`**: Renders the main dashboard user interface.
- **`GET /video_feed`**: Stream endpoint returning continuous multipart JPEG frames.
- **`POST /api/start`**: Accepts `{"source": "<input>"}` payload to start or switch the processing pipeline.
- **`POST /api/stop`**: Halts current video stream processing.
- **`GET /api/metrics`**: Returns current instant detection counts and hourly historical series formatted for Chart.js.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
