# main.py
from slib import ShipDetector, ShipTracker, Painter
import cv2
import threading
import time

# Configuration
MODEL_PATH = '/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
LABEL_PATH = '/home/pi/mind/labelmap.txt'
THRESHOLD = 0.35
VIDEO_PATH = "/home/pi/mind/videos/fregate-move.mp4"

# Global variables
current_frame = None
detections = []
trackers = {}
lock = threading.Lock()

# Initialize detector
detector = ShipDetector(MODEL_PATH, LABEL_PATH, THRESHOLD)
cap = cv2.VideoCapture(VIDEO_PATH)
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def detection_worker():
    global current_frame, detections
    while True:
        with lock:
            if current_frame is not None:
                frame = current_frame.copy()
                dets = detector.process(frame)
                detections = dets if dets else []
        time.sleep(0.1)

def tracking_worker():
    global current_frame, trackers
    while True:
        with lock:
            if current_frame is not None:
                frame = current_frame.copy()
                # Update existing trackers
                to_delete = []
                for tid, tracker in list(trackers.items()):
                    bbox = tracker.update(frame)
                    if bbox:
                        trackers[tid] = bbox
                    else:
                        to_delete.append(tid)
                # Remove lost trackers
                for tid in to_delete:
                    del trackers[tid]
                # Create new trackers for fresh detections
                for det in detections:
                    tid = hash(tuple(det[:4]))
                    if tid not in trackers:
                        try:
                            trackers[tid] = ShipTracker(frame, det[:4]).update(frame)
                        except:
                            pass
        time.sleep(0.05)

# Start threads
threading.Thread(target=detection_worker, daemon=True).start()
threading.Thread(target=tracking_worker, daemon=True).start()

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    with lock:
        current_frame = frame.copy()
    
    # Get latest data
    with lock:
        frame_draw = frame.copy()
        curr_dets = detections
        curr_trks = trackers.copy()
    
    # Draw all elements
    Painter.paint(frame_draw, curr_dets, curr_trks)
    
    cv2.imshow('Boat Tracking', frame_draw)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
