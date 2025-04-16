from slib import ShipDetector, ShipTracker, Painter
import cv2
import threading
import time

detector = ShipDetector(
    TF_LITE_MODEL='/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite',
    LABEL_MAP='/home/pi/mind/labelmap.txt',
    THRESHOLD=0.35
)
painter = Painter()
cap = cv2.VideoCapture("/home/pi/mind/videos/fregate-move.mp4")
trackers = []
current_detections = []
lock = threading.Lock()

def detection_task():
    global current_detections
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        with lock:
            detections = detector.process(frame)
            if detections:
                current_detections = detections
                trackers.clear()
                for d in detections:
                    trackers.append(ShipTracker(frame, d['bbox']))
        time.sleep(0.1)

def tracking_task():
    global current_detections
    while True:
        ret, frame = cap.read()
        if not ret or not trackers:
            continue
        updated = []
        with lock:
            for tracker in trackers:
                bbox = tracker.update(frame)
                if bbox:
                    updated.append({'bbox': bbox, 'score': None})
            current_detections = updated
        time.sleep(0.05)

threading.Thread(target=detection_task, daemon=True).start()
threading.Thread(target=tracking_task, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    with lock:
        if current_detections:
            if trackers:  # Если идет трекинг
                painter.draw_boxes(frame, current_detections, 'track')
            else:  # Если обнаружение
                painter.draw_boxes(frame, current_detections, 'detect')
    
    cv2.imshow('Boat Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
