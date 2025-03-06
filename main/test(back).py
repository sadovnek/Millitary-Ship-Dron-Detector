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

def tracking(original_frame):
    global processed_frame, tracker
    frame = original_frame.copy()
    roi = frame[x1:x1+y1, x2:x2+y2]
    success, roi = tracker.update(frame)
    if success:
        (x,y,w,h) = tuple(map(int,roi))
        cv2.rectangle(frame, (x,y), (x+w, y+h), (100,10,50), 5)
        # Расчет центра отслеживаемого объекта
        center_x = x + w // 2
        center_y = y + h // 2
        # Отрисовка точки в центре отслеживаемого объекта
        cv2.circle(frame, (center_x, center_y), radius=10, color=(0,0,255), thickness=-1)
        # Вывод координат центра отслеживаемого объекта
        print(f"Center of tracked object: ({center_x}, {center_y})")
    else :
        cv2.putText(frame, "Tracking failed", (100,200), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
    

def military(original_frame):
    global processed_frame, state
    frame = original_frame.copy()
    
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
ret = None
x1 = 0; y1 =0; x2 = 0; y2 = 0
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
milmodel = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/aleksandrkrinicyn/Documents/START/networks/yolov5/runs/train/exp/weights/best.pt')
tracker = cv2.TrackerCSRT_create()

def track():
    global state, x, y
    while True:
        with frame_lock:
            local_frame = frame.copy() if ret else None
        if local_frame is not None:
            if state == 2:
                tracking(local_frame)
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
                object(local_frame)
            elif state == 1:
                military(local_frame)
        time.sleep(0.01)

threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

video_path = 'uss.mp4'
cap = cv2.VideoCapture(video_path)
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