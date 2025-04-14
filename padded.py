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
        
    
    def process(self,frame):
        frame_HEIGHT, frame_WIDTH = frame.shape[:2]
        pad = abs(frame_WIDTH - frame_HEIGHT) // 2
        x_pad = pad if frame_HEIGHT > frame_WIDTH else 0
        y_pad = pad if frame_WIDTH > frame_HEIGHT else 0
        frame_padded = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        frame_HEIGHT, frame_WIDTH = frame_padded.shape[:2]
        frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.INPUT_WIDTH, self.INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        input_data = np.expand_dims(frame_resized, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        for score, box, class_ in zip(scores, boxes, classes):
            if score < self.THRESHOLD:
                continue
            coordinates = [0,0,0,0]
            coordinates[0] = round(box[0] * frame_HEIGHT)
            coordinates[1] = round(box[1] * frame_WIDTH)
            coordinates[2] = round(box[2] * frame_HEIGHT)
            coordinates[3] = round(box[3] * frame_WIDTH)
            
            class_name = self.labels[int(class_)]
            if(class_name == 'boat'):
                cv2.rectangle(frame_padded, (coordinates[1], coordinates[0]), (coordinates[3], coordinates[2]), (255, 0, 255), 2)
                frame = frame_padded[y_pad: frame_HEIGHT - y_pad, x_pad: frame_WIDTH - x_pad]
                #return coordinates[1], coordinates[0], coordinates[3], coordinates[2]
                return frame
                
            return -1
