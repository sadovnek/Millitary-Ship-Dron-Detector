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
            y1 = int((y1 - (size-h)/2).clip(0, h)
            x1 = int((x1 - (size-w)/2).clip(0, w)
            y2 = int((y2 - (size-h)/2).clip(0, h)
            x2 = int((x2 - (size-w)/2).clip(0, w))
            
            if (x2 > x1 and y2 > y1) and (x2-x1) > (y2-y1+200):
                detections.append((x1, y1, x2, y2, float(score)))
        
        # NMS
        if detections:
            boxes_nms = [[d[0], d[1], d[2]-d[0], d[3]-d[1]] for d in detections]
            scores_nms = [d[4] for d in detections]
            indices = cv2.dnn.NMSBoxes(boxes_nms, scores_nms, self.threshold, self.nms_threshold)
            if len(indices) > 0:
                detections = [detections[i] for i in indices.flatten()]
        
        return detections

class YOLODetector:
    def __init__(self, model_path, label_path, box_threshold=0.5, class_threshold=0.5, nms_threshold=0.4):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.box_threshold = box_threshold
        self.class_threshold = class_threshold
        self.nms_threshold = nms_threshold  # Исправлено: добавлен параметр NMS

    def process(self, frame):
        h, w = frame.shape[:2]
        size = max(h, w)
        x_pad = (size - w) // 2
        y_pad = (size - h) // 2        
        padded = np.zeros((size, size, 3), dtype=np.uint8)
        padded[y_pad:y_pad+h, x_pad:x_pad+w] = frame
        
        img_resized = cv2.resize(padded, (self.input_shape[1], self.input_shape[0]))
        input_data = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
        
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
            
            # Исправлено: проверка по имени класса
            if class_prob < self.class_threshold or self.labels[class_id] != 'boat':
                continue
            
            # Исправлено: конвертация координат с учетом паддинга
            cx, cy, bw, bh = output[:4]
            cx_abs = cx * size
            cy_abs = cy * size
            bw_abs = bw * size
            bh_abs = bh * size
            
            x_orig = (cx_abs - bw_abs/2) - x_pad
            y_orig = (cy_abs - bh_abs/2) - y_pad
            x1 = max(0, int(x_orig))
            y1 = max(0, int(y_orig))
            x2 = min(w, int(x_orig + bw_abs))
            y2 = min(h, int(y_orig + bh_abs))
            
            # Исправлено: фильтрация по размеру
            if (x2 <= x1 or y2 <= y1) or (x2 - x1) < (y2 - y1):
                continue
            
            boxes.append([x1, y1, x2, y2])
            scores.append(float(box_confidence * class_prob))
            class_ids.append(class_id)
        
        # Исправлено: параметры NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.class_threshold, self.nms_threshold)
        detections = []
        if indices is not None:
            for i in indices.flatten():
                x1, y1, x2, y2 = boxes[i]
                detections.append((
                    x1, y1, x2, y2,
                    scores[i],
                    self.labels[class_ids[i]]
                ))
        return detections

class ShipTracker:
    def __init__(self, ok, frame, result):
        x1, y1, x2, y2, conf, cls = result
        orig_width = x2 - x1
        orig_height = y2 - y1
        
        scale = 0.2
        center_shift_ratio = 0.15
        
        new_width = max(1, int(orig_width * scale))
        new_height = max(1, int(orig_height * scale))
        
        orig_center_x = x1 + orig_width // 2
        orig_center_y = y1 + orig_height // 2
        
        shifted_center_y = orig_center_y + int(orig_height * center_shift_ratio)
        
        new_x = orig_center_x - new_width // 2
        new_y = shifted_center_y - new_height // 2
        
        h, w = frame.shape[:2]
        new_x = max(0, min(new_x, w - new_width))
        new_y = max(0, min(new_y, h - new_height))
        
        self.bbox = (new_x, new_y, new_width, new_height)
        self.tracker = cv2.TrackerCSRT_create()
        self.ok = self.tracker.init(frame, self.bbox)
    
    def update(self, ret, frame):
        self.ok = ret
        self.ok, self.bbox = self.tracker.update(frame)
        x, y, w, h = self.bbox
        return [int(x), int(y), int(x + w), int(y + h)]
