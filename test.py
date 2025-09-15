from ultralytics import YOLO
from kafka import KafkaProducer
import json
import cv2

# 1️⃣ تحميل الموديل
model = YOLO("best.pt")

# 2️⃣ إعداد Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# 3️⃣ فتح الفيديو
cap = cv2.VideoCapture("video.mp4")  # أو 0 لو كاميرا

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 4️⃣ الاستدلال (Detection)
    results = model(frame)

    # 5️⃣ استخراج النتائج المهمّة
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        xyxy = box.xyxy.tolist()[0]
        detections.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": xyxy
        })

    # 6️⃣ إرسال إلى Kafka
    producer.send("yolo-detections", {"frame_id": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                                      "detections": detections})

    # (اختياري) عرض الفيديو محليًا
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
producer.flush()
