#TF_LITE_MODEL = '/home/pi/tensorflow/TF-Lite-Python-Object-Objection/lite-model_yolo-v5-tflite_tflite_model_1.tflite'
TF_LITE_MODEL = "/home/pi/garden/best-s-fp16.tflite"
LABEL_MAP = '/home/pi/tensorflow/TF-Lite-Python-Object-Objection/labelmap.txt'
BOX_THRESHOLD = 0.1
CLASS_THRESHOLD = 0.1
LABEL_SIZE = 1
RUNTIME_ONLY = True

import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter
interpreter = Interpreter(model_path=TF_LITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

cap = cv2.VideoCapture("/home/pi/mind/videos/friends.mp4")
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('filename-3.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 
cnt = 0
while True:
    ret, frame = cap.read()
    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
    pad = round(abs(FRAME_WIDTH - FRAME_HEIGHT) / 2)
    x_pad = pad if FRAME_HEIGHT > FRAME_WIDTH else 0
    y_pad = pad if FRAME_WIDTH > FRAME_HEIGHT else 0
    frame_padded = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    FRAME_HEIGHT, FRAME_WIDTH = frame_padded.shape[:2]
    frame_rgb = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
    input_data = np.expand_dims(frame_resized / 255, axis=0).astype('float32')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])[0]
    boxes = [];box_confidences = [];classes = [];class_probs = []
    
    for output in outputs:
        box_confidence = output[4]
        if box_confidence < BOX_THRESHOLD:
            continue
    
        class_ = output[5:].argmax(axis=0)
        class_prob = output[5:][class_]
    
        if class_prob < CLASS_THRESHOLD:
            continue

        cx, cy, w, h = output[:4] * np.array([FRAME_WIDTH, FRAME_HEIGHT, FRAME_WIDTH, FRAME_HEIGHT])
        x = round(cx - w / 2)
        y = round(cy - h / 2)
        w, h = round(w), round(h)
    
        boxes.append([x, y, w, h])
        box_confidences.append(box_confidence)
        classes.append(class_)
        class_probs.append(class_prob)

    indices = cv2.dnn.NMSBoxes(boxes, box_confidences, BOX_THRESHOLD, BOX_THRESHOLD - 0.1)
    for indice in indices:
        x, y, w, h = boxes[indice]
        class_name = labels[classes[indice]]
        score = box_confidences[indice] * class_probs[indice]
        color = [int(c) for c in colors[classes[indice]]]
        text_color = (255, 255, 255) if sum(color) < 144 * 3 else (0, 0, 0)
        cv2.rectangle(frame_padded, (x, y), (x + w, y + h), color, 4)
		
        label = f'Fregate: {score*100:.2f}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, 7)
        cv2.rectangle(frame_padded,
            (x, y + baseLine), (x + labelSize[0], y - baseLine - labelSize[1]),
            color, cv2.FILLED) 
        cv2.putText(frame_padded, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, text_color, 2,5)
    
    frame_show = frame_padded[y_pad: FRAME_HEIGHT - y_pad, x_pad: FRAME_WIDTH - x_pad]
    cv2.imshow('Object detection', frame_padded)
    #cv2.imwrite(f'./result_yolo_{cnt}.jpg', frame_show)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cnt+=1
    print(x,y,w,h)
    result.write(frame_show)
    print(f'x_pad:{x_pad} y_pad:{y_pad}')

cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
		
		
		
		
		
		
		
		
