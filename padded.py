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
        
        # Получение координат на дополненном изображении
        y1_padded = round(box[0] * padded_height)
        x1_padded = round(box[1] * padded_width)
        y2_padded = round(box[2] * padded_height)
        x2_padded = round(box[3] * padded_width)
        
        # Коррекция координат с учетом паддинга
        x1 = x1_padded - x_pad
        y1 = y1_padded - y_pad
        x2 = x2_padded - x_pad
        y2 = y2_padded - y_pad
        
        # Обрезка до границ исходного изображения
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(original_width, x2)
        y2 = min(original_height, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue  # Пропустить невалидные
        
        detections.append((x1, y1, x2, y2))
    
    return detections if detections else -1
