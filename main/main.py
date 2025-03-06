import torch
import cv2
import time
import os
import random
import threading
from threading import Lock

def object(frame):
    
    with torch.amp.autocast('cuda', dtype=torch.float16):
        results = model(frame)
    
    boats = results.xyxy[0][results.xyxy[0][:, 5] == 8]
    
    if len(boats) > 0:
        print(f"Found {len(boats)} boats")
        for boat in boats:
            x1, y1, x2, y2, conf, cls = boat
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'Boat' #{conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    else:
        print("No boats found")
    
    cv2.imshow('Boat Detection', frame)
    
    
workings = ["detect","type","tracking"]
state = 0
x = 0; y =0
frame = -1
ret = None
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def track():
    global state
    global x,y
    while(True):
        time.sleep(1)
        if(state == 2):
            x = random.randint(0,100)
            y = random.randint(0,100)
            

def type():
    global state
    while(True):
        time.sleep(1)
        if(state == 1):
            dif = random.randint(0,100)
            if dif < 25:
                state = 2
    
def detect():
    global state
    while(True):
        if not ret:
            print("Error: Could not read frame.")
        else:
            object(frame)


threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=type, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

'''main'''
video_path = 'large.mp4'
cap = cv2.VideoCapture(video_path)


while(True):
    time.sleep(0.02)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('Boat Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
