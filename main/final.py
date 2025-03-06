import torch
import cv2
import time
import threading
from threading import Lock

# Добавляем блокировки для синхронизации
frame_lock = Lock()
processed_frame = None
tracker_initialized = False
height = 0
width = 0

def tracking(original_frame,height,width):
    global processed_frame, tracker, tracker_initialized
    frame = original_frame.copy()
    
    if not tracker_initialized:
        # Инициализация трекера при первом вызове
        bbox = (x1, y1, x2 - x1, y2 - y1)  # Пример координат из обнаруженного объекта
        tracker.init(frame, bbox)
        tracker_initialized = True
    
    success, bbox = tracker.update(frame)
    if success:
        x, y, w, h = tuple(map(int, bbox))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 10, 50), 5)
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
        deviation_x = width//2 - center_x
        deviation_y = height//2 - center_y
        cv2.putText(frame, f"delta X: {deviation_x}", (center_x + 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"delta Y: {deviation_y}", (center_x, center_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.circle(frame, (width//2, height//2), radius=10, color=(0,255,0), thickness=-1)
        cv2.line(frame, (center_x, center_y), (center_x, int(height//2)), (0,0,255), 5)
        cv2.line(frame, (center_x, int(height//2)), (int(width//2), int(height//2)), (0,255,0), 5)
    else:
        cv2.putText(frame, "Tracking failed", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    with frame_lock:
        processed_frame = frame
    

def military(original_frame):
    global processed_frame, state, x1, y1, x2, y2, tracker_initialized
    frame = original_frame.copy()
    results = milmodel(frame)
    
    # Фильтрация результатов с уверенностью > 70%
    filtered_results = results.xyxy[0][results.xyxy[0][:, 4] > 0.7]
    
    if len(filtered_results) > 0:
        print(f"Found military {len(filtered_results)} boats")
        # Берем первый обнаруженный корабль с высокой уверенностью
        boat = filtered_results[0]
        x1, y1, x2, y2, _, _ = map(int, boat)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'Military {boat[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        state = 2
        tracker_initialized = False
    else:
        print("No military found")
    with frame_lock:
        processed_frame = frame

def object(original_frame):
    global processed_frame, state
    frame = original_frame.copy()
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        results = model(frame)
    
    # Фильтрация по классу 8 (лодки) и уверенности > 70%
    boats = results.xyxy[0][(results.xyxy[0][:, 5] == 8) & (results.xyxy[0][:, 4] > 0.7)]
    
    if len(boats) > 0:
        print(f"Found {len(boats)} boats")
        for boat in boats:
            x1, y1, x2, y2, conf, cls = boat
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f'Ship {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            state = 1
    else:
        print("No boats found")
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
                tracking(local_frame, int(height), int(width))
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

video_path = 0;
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