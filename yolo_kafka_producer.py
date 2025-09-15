import json
import time
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from kafka import KafkaProducer
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ShelfProducer:
    def __init__(self, model_path='best.pt', kafka_bootstrap='localhost:9092', topic='shelf_removals',
                 video_path='test video/video555.mp4', 
                 iou_threshold=0.1, confidence=0.3):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.model.to('cpu')  # Or 'cuda' for GPU
        logger.info(f"Loaded model with classes: {self.model.names}")
        
        # Kafka producer setup
        self.producer = KafkaProducer(
            bootstrap_servers=[kafka_bootstrap],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all'
        )
        logger.info(f"Kafka producer initialized for topic: {topic}")
        
        # Detection state
        self.prev_detections = []
        self.iou_threshold = iou_threshold
        self.confidence = confidence
        self.class_names = self.model.names
        self.topic = topic
        self.removal_id = 0  # Counter for unique removal IDs
        self.sent_removals = set()  # Track sent removals (class_name, timestamp)
        
        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        logger.info("Producer started. Press 'q' to quit.")

    def iou(self, box1, box2):
        """Calculate Intersection over Union (IOU) between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1b, y1b, x2b, y2b = box2
        xi1 = max(x1, x1b)
        yi1 = max(y1, y1b)
        xi2 = min(x2, x2b)
        yi2 = min(y2, y2b)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2b - x1b) * (y2b - y1b)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def match_detections(self, prev_detections, curr_detections):
        """Match previous detections to current; return unmatched prev as removals."""
        unmatched_prev = []
        for prev_det in prev_detections:
            matched = False
            for curr_det in curr_detections:
                if (prev_det['class_name'] == curr_det['class_name'] and 
                    self.iou(prev_det['bbox'], curr_det['bbox']) > self.iou_threshold):
                    matched = True
                    break
            if not matched:
                unmatched_prev.append(prev_det)
        logger.debug(f"Matched {len(prev_detections) - len(unmatched_prev)} detections; {len(unmatched_prev)} potential removals")
        return unmatched_prev

    def draw_detections(self, frame, results):
        """Custom drawing for labels above boxes."""
        annotated_frame = frame.copy()
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                class_id = int(classes[i])
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{self.class_names[class_id]}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                text_offset_y = 10
                text_x = x1
                text_y = y1 - text_offset_y
                
                cv2.rectangle(annotated_frame, (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0], text_y + 5), (0, 0, 0), -1)
                cv2.putText(annotated_frame, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        
        return annotated_frame

    def detect_and_compute_removals(self, frame):
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.confidence)
        curr_detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                class_name = self.class_names[classes[i]]
                bbox = tuple(map(float, boxes[i]))  # (x1, y1, x2, y2)
                curr_detections.append({'class_name': class_name, 'bbox': bbox})
        
        logger.debug(f"Detected {len(curr_detections)} objects in current frame")
        
        # Compute removals
        removals = []
        if self.prev_detections:
            removals = self.match_detections(self.prev_detections, curr_detections)
        
        # De-duplicate based on class_name and timestamp (within 5 seconds)
        new_removals = []
        current_time = time.time()
        for removal in removals:
            key = (removal['class_name'], removal['timestamp'] if 'timestamp' in removal else current_time)
            if key not in self.sent_removals and (current_time - (removal.get('timestamp', current_time))) < 5:
                self.removal_id += 1
                removal['track_id'] = self.removal_id
                removal['timestamp'] = current_time
                new_removals.append(removal)
                self.sent_removals.add(key)
                logger.debug(f"New removal added: {removal}")
        
        # Update previous for next frame
        self.prev_detections = curr_detections
        
        annotated_frame = self.draw_detections(frame, results)
        return annotated_frame, new_removals

    def send_removals(self, removals):
        if not removals:
            logger.debug("No removals to send.")
            return
        try:
            for removal in removals:
                self.producer.send(self.topic, value=removal)
                latency = (time.time() - removal['timestamp']) * 1000
                logger.info(f"Sent removal: {removal} (Latency: {latency:.2f}ms)")
            self.producer.flush()
        except Exception as e:
            logger.error(f"Failed to send removal: {e}")

    def run(self):
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Video ended or failed to read frame.")
                break
            frame_count += 1
            start_time = time.time()
            annotated_frame, removals = self.detect_and_compute_removals(frame)
            if removals:
                self.send_removals(removals)
            else:
                logger.debug(f"No removals detected in frame {frame_count}")
            cv2.imshow('Shelf Monitor - Producer', annotated_frame)
            logger.info(f"Frame {frame_count} processed in {(time.time() - start_time)*1000:.2f}ms")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
        self.producer.close()

if __name__ == "__main__":
    producer = ShelfProducer()
    producer.run()