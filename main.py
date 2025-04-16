from slib import ShipDetector
from slib import ShipTracker
from slib import painter
import threading
import time
import cv2

Detector = ShipDetector(
    TF_LITE_MODEL = '/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite',
    LABEL_MAP = '/home/pi/mind/labelmap.txt',
    THRESHOLD = 0.35
)

draw = painter()
state = -1
video_path = "/home/pi/mind/videos/fregate-move.mp4";
cap = cv2.VideoCapture(video_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
result = [0,0,0,0]
flag = -1
conf = []
print("H" + ":" + str(height) + " " + "W" + ":" + str(width))
ret, frame = cap.read()

def find():
    global state, result
    while True:
        if state == 0:
            result = Detector.process(frame)
            if result != -1:
                state = 1
                Tracker = ShipTracker(frame,result,width,height)
        time.sleep(0.1)

def pathing():
    global state, result
    time.sleep(1)
    state = 0
    while True:
        if state == 1:
            Tracker.tracking(frame,result)
            
        time.sleep(0.1)

threading.Thread(target=find, daemon=True).start()
threading.Thread(target=pathing, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    draw.paint(frame,state,result[0],result[1],result[2],result[3])
    cv2.imshow('Boat Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
print(len(conf))
print(max(conf))
conf = sorted(conf)
conf = conf[::-1]
print(conf)
