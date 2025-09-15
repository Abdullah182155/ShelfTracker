
# Retail Shelf Monitoring System

## Overview
This project is a real-time retail shelf monitoring system that uses YOLOv8 for object detection and Apache Kafka for event streaming. It detects 14 specific product classes on a food store shelf, tracks removals (e.g., when a customer takes a product), and sends low-latency alerts via Kafka. The system processes video feeds (live camera or file) and displays real-time removal notifications.

**Key features:**
- **Object Detection**: YOLOv8 model fine-tuned for 14 product classes (e.g., `Dutch Mill-High Protein Blue-`, `Hooray-Protein shake chocolate milk-`).
- **Removal Detection**: IOU-based frame-to-frame matching with de-duplication, stability checks, and cooldowns to handle occlusions and false positives.
- **Messaging**: Kafka producer sends JSON events:
```json
{"track_id": 1, "class_name": "Hooray-Protein shake chocolate milk-", "timestamp": 1757956288.12}
```
to the `shelf_removals` topic.
- **Visualization**: Annotated video feed with bounding boxes and labels (non-overlapping).
- **Efficiency**: Optimized for ~15-30 FPS, with GPU support.

## Architecture
```
[Video Feed (Camera/Video File)] --> [YOLO Detection + IOU Matching] --> [Removal Logic (De-dupe + Stability)] --> [Kafka Producer] --> [Topic: shelf_removals]
                                                                |
                                                                v
                                                        [Kafka Consumer] --> [Real-Time Alerts (Console/GUI)]
```

## Prerequisites
- Python 3.10+ (Conda environment recommended: `cv`).
- NVIDIA GPU (optional, for faster inference).
- Docker for Kafka.

## Setup

1. **Clone the Repo**
```bash
git clone <repo-url>
cd Retail-Shelf-Monitoring
```

2. **Create Environment**
```bash
conda create -n cv python=3.10
conda activate cv
pip install ultralytics kafka-python opencv-python numpy
```

3. **Model Preparation**
- Download or train `shelf_yolo.pt` (fine-tuned YOLOv8n on your 14 classes).
- Example training:
```bash
yolo task=detect mode=train model=yolov8n.pt data=your_dataset.yaml epochs=100 imgsz=640
```

4. **Kafka Setup** (via Docker)
```bash
docker run -d --name kafka -p 9092:9092 apache/kafka:3.8.0
docker exec kafka /opt/kafka/bin/kafka-topics.sh --create --topic shelf_removals --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```

## Usage

1. **Run Consumer** (in one terminal)
```bash
conda activate cv
python yolo_kafka_consumer.py
```
Output:  
`Consumer started. Listening for removals...`

2. **Run Producer** (in another terminal)
```bash
conda activate cv
python yolo_kafka_producer.py
```
- A window shows the annotated video feed.
- Press `q` to quit.
- For live camera: Edit `video_path` to `0` in `__init__`.

3. **Test Removals**
- Use a video with product removals or a live feed where you remove an item.
- Consumer prints:  
`[REAL-TIME ALERT] Product 'Hooray-Protein shake chocolate milk-' (Track ID: 1) removed at 1757956288.12`.

## Configuration
Edit parameters in `yolo_kafka_producer.py`:
- `confidence=0.2`: Detection threshold.
- `iou_threshold=0.5`: Matching overlap.
- `stability_frames=3`: Confirm removals over N frames.
- `cooldown_seconds=2.0`: Min time between same-class removals.

## Files
- `yolo_kafka_producer.py`: Detection and Kafka producer.
- `yolo_kafka_consumer.py`: Message consumer and alerts.
- `shelf_yolo.pt`: YOLO model weights.
- `test video/PixVerse_V5_Image_Text_360P_Using_the_provided.mp4`: Sample video.

## Troubleshooting
- **No Detections**: Lower `confidence`, verify model with `yolo val`.
- **No Removals**: Ensure video shows changes; test with live feed.
- **Kafka Issues**: Restart Docker container; check topic with `kafka-topics.sh --list`.
- **Duplicates**: Adjust `cooldown_seconds` or `stability_frames`.
- **Performance**: Use GPU (`self.model.to('cuda')`); resize frames to 320x320.

## Enhancements
- GUI Dashboard: Integrate Tkinter/PyQt for consumer.
- Database Sync: Log removals to SQLite/PostgreSQL.
- Multi-Camera: Run multiple producers with unique topics.
- Alerts: Email/Slack notifications on removal.

## License
MIT License. See LICENSE file.

## Contact
For issues, open a GitHub issue or contact [your-email].

---

Last updated: September 15, 2025
