import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class ShipDetector:
    
    def __init__(self, model_path, label_path, threshold, nms_threshold=0.4):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.threshold = threshold
        self.nms_threshold = nms_threshold

    def process(self, frame):
        h, w = frame.shape[:2]
        size = max(h, w)
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[(size-h)//2 : (size-h)//2 + h, (size-w)//2 : (size-w)//2 + w] = frame
        
        img = cv2.resize(padded, self.input_shape)
        img = np.expand_dims(img, axis=0).astype(np.uint8)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        
        detections = []
        for score, box, class_id in zip(scores, boxes, classes):
            if score < self.threshold or self.labels[int(class_id)] != 'boat':
                continue
            
            y1, x1, y2, x2 = box * size
            y1 = int((y1 - (size-h)/2).clip(0, h))
            x1 = int((x1 - (size-w)/2).clip(0, w))
            y2 = int((y2 - (size-h)/2).clip(0, h))
            x2 = int((x2 - (size-w)/2).clip(0, w))
            
            if (x2 > x1 and y2 > y1) and (x2-x1) > (y2-y1+200):
                detections.append((x1, y1, x2, y2, float(score)))
        
        # NMS фильтрация
        if detections:
            boxes_nms = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections]
            scores_nms = [d[4] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.threshold, self.nms_threshold)
            if len(indices) > 0:
                detections = [detections[i] for i in indices.flatten()]
        
        return detections


class YOLODetector:
    def __init__(self, model_path, label_path, box_threshold=0.2, class_threshold=0.25):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]  # (height, width)
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.box_threshold = box_threshold
        self.class_threshold = class_threshold

    def process(self, frame):
        h, w = frame.shape[:2]
        size = max(h, w)
        x_pad = (size - w) // 2
        y_pad = (size - h) // 2        
        # Добавление паддинга
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[y_pad:y_pad+h, x_pad:x_pad+w] = frame
        # Препроцессинг
        img_resized = cv2.resize(padded, (self.input_shape[1], self.input_shape[0]))
        input_data = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
        # Инференс
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        boxes = []
        scores = []
        class_ids = []
        
        for output in outputs:
            box_confidence = output[4]
            if box_confidence < self.box_threshold:
                continue
            
            class_scores = output[5:]
            class_id = np.argmax(class_scores)
            class_prob = class_scores[class_id]
            
            if class_prob < self.class_threshold or class_id != 1:
                continue
            
            # Конвертация координат
            cx, cy, bw, bh = output[:4] * np.array([size, size, size, size])
            x = int(cx - bw/2)
            y = int(cy - bh/2)
            x_orig = x - x_pad
            y_orig = y - y_pad
            x1, y1 = max(0, x_orig), max(0, y_orig)
            x2, y2 = min(w, x_orig + int(bw)), min(h, y_orig + int(bh))
            
            if (x2 <= x1 or y2 <= y1) or (x2 - x1) < (y2 - y1):
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box_confidence * class_prob))
            class_ids.append(class_id)
        
        # Применение NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.box_threshold, self.box_threshold - 0.1)
        detections = []
        for i in indices:
            idx = i if isinstance(indices, np.ndarray) else i[0]
            x1, y1, x2, y2 = boxes[idx]
            detections.append((
                x1, y1, x2, y2,
                scores[idx],
                self.labels[class_ids[idx]]
            ))
        
        return detections

class ShipTracker:
    
    def __init__(self, ok, frame, result):
        # Исходные координаты в формате X1, Y1, X2, Y2
        x1, y1, x2, y2 = result
        orig_width = x2 - x1
        orig_height = y2 - y1
        
        # Коэффициенты
        scale = 0.2  # Уменьшение размера на 80%
        center_shift_ratio = 0.1  # Смещение центра вниз на 10% высоты
        
        # Уменьшенные размеры
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))
        
        # Исходный центр
        orig_center_x = x1 + orig_width // 2
        orig_center_y = y1 + orig_height // 2
        
        # Смещение центра вниз
        shifted_center_y = orig_center_y + int(orig_height * center_shift_ratio)
        
        # Новый верхний левый угол
        new_x = orig_center_x - new_width // 2
        new_y = shifted_center_y - new_height // 2
        
        # Проверка границ кадра
        h, w = frame.shape[:2]
        new_x = max(0, min(new_x, w - new_width))
        new_y = max(0, min(new_y, h - new_height))
        
        self.bbox = (new_x, new_y, new_width, new_height)
        self.tracker = cv2.TrackerKCF_create()
        self.ok = self.tracker.init(frame, self.bbox)
    
    def update(self, ret, frame):
        self.ok = ret
        self.ok, self.bbox = self.tracker.update(frame)
        
        # Возвращаем в формате X1, Y1, X2, Y2
        x, y, w, h = self.bbox
        return [int(x), int(y), int(x + w), int(y + h)]
        
