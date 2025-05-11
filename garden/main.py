import cv2
import numpy as np
import time
from slib import ShipTracker
from slib import ShipDetector
from slib import YOLODetector
import threading

# Конфигурация
#MODEL_PATH = "/home/pi/garden/best-r-fp16.tflite"
LABEL_PATH = "/home/pi/garden/label.txt"
THRESHOLD = 0.1
VIDEO_SOURCE = 0#"/home/pi/mind/videos/friends-1.mov"

# Общие ресурсы
cap = cv2.VideoCapture(VIDEO_SOURCE)
state = -1
racer = -1
track = None  # Global tracker instance
bbox = [0, 0, 0, 0]  # Initial bbox
vision = [0,0,0,0]

'''
detector = ShipDetector(
    model_path=MODEL_PATH,
    label_path=LABEL_PATH,
    threshold=THRESHOLD,
)
'''

detector = YOLODetector(
    model_path="/home/pi/garden/best-l-fp16.tflite",
    label_path="/home/pi/garden/label.txt",
)

results = [0, 0, 0, 0]

def find():
    global results, state, bbox, track, vision, racer
    time.sleep(1)
    while True:
        if state == 0:
            results = detector.process(frame)
        elif state == 1:
            vision = racer.update(ret,frame)
            
def draw():
    global frame, state, bbox, track, racer
    if state == 0 and results and results != [0, 0, 0, 0]:
        for ship in results:
            cv2.putText(frame, f'Military {ship[4]:.2f}', (ship[0], ship[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (ship[0], ship[1]), (ship[2], ship[3]), (71, 255, 0), 2)
            if ship[4] > 0.75:
                racer = ShipTracker(ret,frame,ship)
                state = 1
    elif state == 1:
        cv2.rectangle(frame, (vision[0],vision[1]), (vision[2],vision[3]), (0,0,255), 2, 1)
        
threading.Thread(target=find, daemon=True).start()
state = 0  # Start with detection mode

while True:
    ret, frame = cap.read()
    if not ret:
        break

    draw()
    print(results, f"state: {state}")
    cv2.imshow('Ship Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    #time.sleep(0.01)

cap.release()
cv2.destroyAllWindows()
