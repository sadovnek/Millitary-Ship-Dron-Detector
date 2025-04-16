import cv2
import time
import threading
from threading import Lock
from slib import ShipDetector, ShipTracker

# Блокировки и общие ресурсы
frame_lock = Lock()
current_frame = None
tracker = None
result = -1
height = 0
width = 0
state = 0  # 0 - detection, 1 - tracking

# Инициализация детектора
detector = ShipDetector(
    TF_LITE_MODEL='/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite',
    LABEL_MAP='/home/pi/mind/labelmap.txt',
    THRESHOLD=0.15
)

def track():
    global state, tracker, current_frame
    while True:
        with frame_lock:
            local_frame = current_frame.copy() if current_frame is not None else None
        
        if local_frame is not None and state == 1 and tracker is not None:
            success, bbox = tracker.tracking(local_frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                print("TRACKING")
                cv2.rectangle(current_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        time.sleep(0.01)

def detect():
    global state, tracker, current_frame, result
    while True:
        with frame_lock:
            local_frame = current_frame.copy() if current_frame is not None else None
        
        if local_frame is not None and state == 0:
            result = detector.process(local_frame)
            if result and result != -1:
                x1, y1, x2, y2 = result
                w = x2 - x1
                h = y2 - y1
                with frame_lock:
                    tracker = ShipTracker(local_frame, (x1, y1, w, h))
                    state = 1  # Переключаем в режим трекинга
        
        time.sleep(0.01)

# Запуск потоков
threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

# Инициализация видео
video_path = "videos/fregate-move.mp4"
cap = cv2.VideoCapture(video_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Video resolution: {width}x{height}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    with frame_lock:
        current_frame = frame.copy()

    # Отображение кадра с обработкой
    display_frame = current_frame.copy()
    print(result,state)
    cv2.imshow('Boat Detection', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
