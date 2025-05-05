import cv2
import time
import threading
import numpy as np
import tensorflow as tf
from threading import Lock

# Блокировки и глобальные переменные
frame_lock = Lock()
processed_frame = None
tracker_initialized = False
height, width = 0, 0

# Инициализация TFLite моделей
model_path = '/Users/aleksandrkrinicyn/Documents/START/networks/yolov5/yolov5s-fp16.tflite'
mil_model_path = 'networks/yolov5/runs/train/exp/weights/best-fp16.tflite'

# Загрузка моделей
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
mil_interpreter = tf.lite.Interpreter(model_path=mil_model_path)
mil_interpreter.allocate_tensors()

# Получение информации о входах/выходах
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
mil_input_details = mil_interpreter.get_input_details()
mil_output_details = mil_interpreter.get_output_details()

# Функции предобработки изображения
def preprocess(img, interpreter):
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_size = input_shape[1]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def postprocess(outputs, frame_shape, conf_thresh=0.7):
    outputs = np.squeeze(outputs)
    boxes = outputs[:, :4]
    scores = outputs[:, 4]
    classes = outputs[:, 5]
    
    # Масштабирование координат к исходному размеру кадра
    scale = max(frame_shape) / outputs.shape[0]
    boxes *= scale
    
    valid = scores > conf_thresh
    return boxes[valid], scores[valid], classes[valid]

def tracking(original_frame, height, width):
    global processed_frame, tracker, tracker_initialized
    frame = original_frame.copy()
    
    if not tracker_initialized:
        bbox = (x1, y1, x2 - x1, y2 - y1)
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
        cv2.putText(frame, f"delta X: {deviation_x}", (center_x + 50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"delta Y: {deviation_y}", (center_x, center_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.circle(frame, (width//2, height//2), 10, (0,255,0), -1)
        cv2.line(frame, (center_x, center_y), (center_x, height//2), (0,0,255), 5)
        cv2.line(frame, (center_x, height//2), (width//2, height//2), (0,255,0), 5)
    else:
        cv2.putText(frame, "Tracking failed", (100, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    with frame_lock:
        processed_frame = frame

def military(original_frame):
    global processed_frame, state, x1, y1, x2, y2, tracker_initialized
    frame = original_frame.copy()
    
    input_data = preprocess(frame, mil_interpreter)
    mil_interpreter.set_tensor(mil_input_details[0]['index'], input_data)
    mil_interpreter.invoke()
    outputs = mil_interpreter.get_tensor(mil_output_details[0]['index'])
    
    boxes, scores, classes = postprocess(outputs, frame.shape)
    
    if len(boxes) > 0:
        print(f"Found military {len(boxes)} boats")
        x1, y1, x2, y2 = boxes[0].astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f'Military {scores[0]:.2f}', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        state = 2
        tracker_initialized = False
    else:
        print("No military found")
    
    with frame_lock:
        processed_frame = frame

def object(original_frame):
    global processed_frame, state
    frame = original_frame.copy()
    
    input_data = preprocess(frame, interpreter)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])
    
    boxes, scores, classes = postprocess(outputs, frame.shape)
    
    boats_mask = (classes == 8)
    boats = boxes[boats_mask]
    
    if len(boats) > 0:
        print(f"Found {len(boats)} boats")
        for box in boats:
            x1, y1, x2, y2 = box.astype(int)
            label = f'Ship {scores[0]:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            state = 1
    
    with frame_lock:
        processed_frame = frame

# Глобальные переменные состояния
state = 0
ret = None
x1, y1, x2, y2 = 0, 0, 0, 0
tracker = cv2.TrackerCSRT_create()

def track():
    global state
    while True:
        with frame_lock:
            local_frame = frame.copy() if ret else None
        if local_frame is not None and state == 2:
            tracking(local_frame, height, width)
        time.sleep(0.01)

def detect():
    global state
    while True:
        with frame_lock:
            local_frame = frame.copy() if ret else None
        if local_frame is not None:
            if state == 0:
                object(local_frame)
            elif state == 1:
                military(local_frame)
        time.sleep(0.01)

# Запуск потоков
threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

# Инициализация видеопотока
video_path = "large.mp4"
cap = cv2.VideoCapture(video_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"Resolution: {width}x{height}")

while True:
    with frame_lock:
        ret, frame = cap.read()
        if not ret:
            break
        
        if processed_frame is not None:
            cv2.imshow('Boat Detection', processed_frame)
            print(state)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1/30)
    
cap.release()
cv2.destroyAllWindows()