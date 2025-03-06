import torch
import cv2
import time
import os
import random
import threading
from threading import Lock

# Добавляем блокировки для синхронизации
frame_lock = Lock()
processed_frame = None
ret = None

def tracking(original_frame):
    global processed_frame
    frame = original_frame.copy()

def military(original_frame):
    global processed_frame
    frame = original_frame.copy()
    # Используем специализированную модель для военных кораблей
    results = milmodel(frame)
    if len(results.xyxy[0]) > 0:
        print(f"Found military {len(results.xyxy[0])} boats")
        for boat in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = boat
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Красный цвет для военных
            cv2.putText(frame, 'Military', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            state = 2
    else:
        print("No military found")
    with frame_lock:
        processed_frame = frame


def object(original_frame):
    global processed_frame, state
    frame = original_frame.copy()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        results = model(frame)
    boats = results.xyxy[0][results.xyxy[0][:, 5] == 8]
    if len(boats) > 0:
        print(f"Found {len(boats)} boats")
        for boat in boats:
            x1, y1, x2, y2, conf, cls = boat
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'Boat' 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            state = 1
    else:
        print("No boats found")
    # Сохраняем обработанный кадр с блокировкой
    with frame_lock:
        processed_frame = frame

workings = ["detect","type","tracking"]
state = 0
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
milmodel = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/aleksandrkrinicyn/Documents/START/networks/yolov5/runs/train/exp/weights/best.pt')
tracker = cv2.TrackerCSRT_create()

def track():
    global state, x, y
    while True:
        if state == 2:
            x = random.randint(0,100)
            y = random.randint(0,100)
        else:
            pass
    
def detect():
    while True:
        if state != 0:
            pass
        else:
            with frame_lock:
                local_frame = frame.copy() if ret else None
            if local_frame is not None:
                object(local_frame)
            #time.sleep(0.01)

threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

video_path = 'uss.mp4'
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

while True:
    #time.sleep(0.03)
    # Чтение кадра с блокировкой
    with frame_lock:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Отображение последнего обработанного кадра
        if processed_frame is not None:
            cv2.imshow('Boat Detection', processed_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()