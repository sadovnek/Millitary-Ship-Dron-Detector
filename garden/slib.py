import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class ShipDetector:
    
    def __init__(self, model_path, label_path, threshold=0.25, nms_threshold=0.4):
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

    def __init__(self,TF_LITE_MODEL, LABEL_PATH, BOX_THRESHOLD, CLASS_THRESHOLD):
        ''' Initilization model parametrs '''
        self.interpreter = Interpreter(model_path=TF_LITE_MODEL)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.height, self.width, _ = self.interpreter.get_input_details()[0]['shape']
        with open(LABEL_PATH, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        self.TF_LITE_MODEL = TF_LITE_MODEL
        self.LABEL_PATH = LABEL_PATH
        self.BOX_THRESHOLD = BOX_THRESHOLD
        self.CLASS_THRESHOLD = CLASS_THRESHOLD
        
        
    def process(self,frame_padded):
        ''' Work with frame(image)'''
        FRAME_HEIGHT, FRAME_WIDTH = frame_padded.shape[:2]
        frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(frame_resized / 255, axis=0).astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        boxes = [];box_confidences = [];classes = [];class_probs = []
        for output in outputs:
            box_confidence = output[4]
            if box_confidence < self.BOX_THRESHOLD:
                continue
            class_ = output[5:].argmax(axis=0)
            class_prob = output[5:][class_]
            if class_prob < self.CLASS_THRESHOLD:
                continue
            cx, cy, w, h = output[:4] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])
            x = round(cx - w / 2)
            y = round(cy - h / 2)
            w, h = round(w), round(h)
            boxes.append([x, y, w, h])
            box_confidences.append(box_confidence)
            classes.append(class_)
            class_probs.append(class_prob)
            
        indices = cv2.dnn.NMSBoxes(boxes, box_confidences, self.BOX_THRESHOLD, self.BOX_THRESHOLD - 0.1)
        bbox = []
        for indice in indices:
            x, y, w, h = boxes[indice]
            class_name = self.labels[classes[indice]]
            score = box_confidences[indice] * class_probs[indice]
            bbox.append((x,y,w,h,score,class_name))
        return bbox
            
        
