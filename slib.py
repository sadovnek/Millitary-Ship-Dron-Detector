import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

class ShipDetector():
    
    def __init__(self,TF_LITE_MODEL,LABEL_MAP,THRESHOLD):
        self.TF_LITE_MODEL = TF_LITE_MODEL
        self.LABEL_MAP = LABEL_MAP
        self.THRESHOLD = THRESHOLD
        self.interpreter = Interpreter(model_path=TF_LITE_MODEL)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        _, self.INPUT_HEIGHT, self.INPUT_WIDTH, _ = self.interpreter.get_input_details()[0]['shape']
        with open(LABEL_MAP, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
def process(self, frame):
    original_height, original_width = frame.shape[:2]
    pad = abs(original_width - original_height) // 2
    x_pad = pad if original_height > original_width else 0
    y_pad = pad if original_width > original_height else 0
    frame_padded = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_height, padded_width = frame_padded.shape[:2]

    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (self.INPUT_WIDTH, self.INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(frame_resized, axis=0)

    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
    self.interpreter.invoke()

    boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
    classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
    scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

    detections = []
    for score, box, class_ in zip(scores, boxes, classes):
        if score < self.THRESHOLD:
            continue
            
        class_name = self.labels[int(class_)]
        if class_name != 'boat':
            continue
    
        y1_padded = round(box[0] * padded_height)
        x1_padded = round(box[1] * padded_width)
        y2_padded = round(box[2] * padded_height)
        x2_padded = round(box[3] * padded_width)
        
        x1 = x1_padded - x_pad
        y1 = y1_padded - y_pad
        x2 = x2_padded - x_pad
        y2 = y2_padded - y_pad
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width, x2)
        y2 = min(original_height, y2)
    
        if x1 >= x2 or y1 >= y2:
            continue
    
        # Добавляем координаты в виде кортежа и confidence score
        detections.append((x1, y1, x2, y2, float(score)))
    
    # Возвращаем список всех детекций или пустой список
    return detections if detections else -1

