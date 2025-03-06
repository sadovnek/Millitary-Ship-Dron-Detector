import torch
import cv2
import numpy as np
import os
import time

def save(frame):
    cv2.imwrite(f"bufer/{time.time()}.jpg", frame)

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Проверка классов, используемых в модели
print("Model classes:", model.names)

# Открытие видеопотока (0 для веб-камеры, или путь к видеофайлу)
video_path = 'large.mp4' # или 'path_to_your_video.mp4'
cap = cv2.VideoCapture(video_path)

# Проверка успешности открытия видеопотока
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()

    # Если кадр не прочитан, завершаем цикл
    if not ret:
        print("Error: Could not read frame.")
        break

    # Распознавание объектов на кадре
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        results = model(frame)

    # Вывод результатов для отладки
    print("Detection results:", results.xyxy[0])

    # Фильтрация результатов для лодок (класс 8 в COCO dataset)
    boats = results.xyxy[0][results.xyxy[0][:, 5] == 8]

    # Проверка, найдены ли лодки
    if len(boats) > 0:
        print(f"Found {len(boats)} boats")
    else:
        print("No boats found")

    # Отображение результатов
    for boat in boats:
        x1, y1, x2, y2, conf, cls = boat
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f'Boat' #{conf:.2f}'
        save(frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Отображение кадра с распознанными лодками
    cv2.imshow('Boat Detection', frame)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
