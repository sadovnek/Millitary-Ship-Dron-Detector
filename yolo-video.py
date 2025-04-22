TF_LITE_MODEL = './lite-model_yolo-v5-tflite_tflite_model_1.tflite'
LABEL_MAP = './labelmap.txt'
BOX_THRESHOLD = 0.5
CLASS_THRESHOLD = 0.5
LABEL_SIZE = 0.5

import cv2
import numpy as np
import tensorflow as tf

# Инициализация модели
interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, height, width, _ = input_details[0]['shape']

# Загрузка меток
with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Инициализация видеопотока (0 - веб-камера)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Предобработка кадра
    IMG_HEIGHT, IMG_WIDTH = frame.shape[:2]
    
    # Добавление паддинга для квадрата
    pad = round(abs(IMG_WIDTH - IMG_HEIGHT) / 2)
    x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
    y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
    img_padded = cv2.copyMakeBorder(frame, y_pad, y_pad, x_pad, x_pad, 
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    # Преобразование для модели
    img_resized = cv2.resize(img_padded, (width, height))
    input_data = np.expand_dims(img_resized / 255, axis=0).astype('float32')
    
    # Обработка моделью
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    outputs = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Постобработка
    boxes = []
    confidences = []
    class_ids = []
    
    for output in outputs:
        confidence = output[4]
        if confidence < BOX_THRESHOLD:
            continue
            
        class_id = output[5:].argmax()
        class_prob = output[5:][class_id]
        
        if class_prob < CLASS_THRESHOLD:
            continue
            
        # Координаты с учетом паддинга
        cx, cy, w, h = output[:4] * [img_padded.shape[1], img_padded.shape[0]] * 2
        x = int(cx - w//2)
        y = int(cy - h//2)
        w, h = int(w), int(h)
        
        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
    
    # NMS подавление
    indices = cv2.dnn.NMSBoxes(boxes, confidences, BOX_THRESHOLD, 0.4)
    
    # Отрисовка боксов
    for i in indices:
        x, y, w, h = boxes[i]
        label = f"{labels[class_ids[i]]} {confidences[i]:.2f}"
        color = colors[class_ids[i]].tolist()
        
        cv2.rectangle(img_padded, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_padded, label, (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, LABEL_SIZE, color, 2)
    
    # Удаление паддинга для отображения
    result_frame = img_padded[y_pad:y_pad+IMG_HEIGHT, x_pad:x_pad+IMG_WIDTH]
    
    # Показ кадра
    cv2.imshow('Video Detection', result_frame)
    
    # Выход по 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()