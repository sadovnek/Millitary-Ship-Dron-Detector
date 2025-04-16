import cv2
import time
import threading
from threading import Lock
from slib import ShipDetector
from slib import ShipTracker

# Добавляем блокировки для синхронизации
frame_lock = Lock()
height = 0
width = 0

Detector = ShipDetector(
    TF_LITE_MODEL = '/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite',
    LABEL_MAP = '/home/pi/mind/labelmap.txt',
    THRESHOLD = 0.25
)

workings = ["detect","type","tracking"]

state = 0
ret = None
x1 = 0; y1 =0; x2 = 0; y2 = 0

def track():
    global state, x, y
    while True:
        with frame_lock:
            local_frame = frame.copy() if ret else None
        if local_frame is not None:
            if state == 2:
                Tracker.tracking(frame,result)
            else:
                pass
        time.sleep(0.01)
            

def detect():
    global mil
    while True:
        with frame_lock:
            local_frame = frame.copy() if ret else None
        if local_frame is not None:
            if state == 0:
                result = Detector.process(frame)
                if result != -1:
                    state = 2
                    Tracker = ShipTracker(frame,result,width,height)
                #object(local_frame)
        time.sleep(0.01)

threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

video_path = "videos/mil-test.mp4";
cap = cv2.VideoCapture(video_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("H" + ":" + str(height) + " " + "W" + ":" + str(width))
ret, frame = cap.read()

while True:
    # Чтение кадра с блокировкой
    with frame_lock:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Отображение последнего обработанного кадра
        if processed_frame is not None:
            cv2.imshow('Boat Detection', processed_frame)
            print(state)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.01)
    
cap.release()
cv2.destroyAllWindows()
