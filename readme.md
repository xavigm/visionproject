# 🛰️ Real-Time Computer Vision & Detection Analytics Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8%2F12.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)
![Supervision](https://img.shields.io/badge/Roboflow-Supervision-purple?style=for-the-badge)
![NVIDIA](https://img.shields.io/badge/NVIDIA-RTX_3060_12GB-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

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
