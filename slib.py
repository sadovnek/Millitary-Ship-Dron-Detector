import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class ShipDetector:
    def __init__(self, model_path, label_path, threshold):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.threshold = threshold

    def process(self, frame):
        h, w = frame.shape[:2]
        # Добавление паддинга для квадрата
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[(size-h)//2 : (size-h)//2 + h, (size-w)//2 : (size-w)//2 + w] = frame
        # Препроцессинг
        img = cv2.resize(padded, self.input_shape)
        img = np.expand_dims(img, axis=0).astype(np.uint8)
        # Инференс
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        # Постобработка
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        # Фильтрация результатов
        detections = []
        for score, box, class_id in zip(scores, boxes, classes):
            if score < self.threshold or self.labels[int(class_id)] != 'boat':
                continue
            # Конвертация координат
            y1, x1, y2, x2 = box * size
            y1 = int((y1 - (size-h)/2).clip(0, h))
            x1 = int((x1 - (size-w)/2).clip(0, w))
            y2 = int((y2 - (size-h)/2).clip(0, h))
            x2 = int((x2 - (size-w)/2).clip(0, w))
            if (x2 > x1 and y2 > y1) and (x2-x1 > y2-y1):
                detections.append((x1, y1, x2, y2, float(score)))
        return detections
