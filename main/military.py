import cv2
import torch

def process_image(image):
    results = model(image)
    results.render()
    # Анализ результатов детекции
    detections = results.pandas().xyxy[0]
    # Определение класса корабля
    type = 'None'  # Значение по умолчанию
    if not detections.empty:
        # Берем первый обнаруженный класс (можно модифицировать логику)
        type = detections['name'].iloc[0] 
    '''# Сохраняем класс в файл
    with open('ship_class.txt', 'w') as f:
        f.write(ship_class)'''
    print(type)
    return results.ims[0] if results.ims else image

# Загрузка модели
model_path = '/Users/aleksandrkrinicyn/Documents/START/networks/yolov5/runs/train/exp/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Загрузка изображения
image_path = 'base.jpg'  # Укажите путь к вашему изображению
image = cv2.imread(image_path)

if image is None:
    print(f"Ошибка: Не удалось загрузить изображение {image_path}")
    exit()

# Обработка изображения
processed_image = process_image(image)

# Сохранение результата
output_path = 'result.jpg'
cv2.imwrite(output_path, processed_image)
print(f"Результат сохранён в {output_path}")

# Показать результат в окне (опционально)
cv2.imshow('Detection Result', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()