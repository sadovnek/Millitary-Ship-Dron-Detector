import cv2
from slib import ShipDetector

def main():
    # Пути к модели и label map (замените на свои)
    MODEL_PATH = "/home/pi/mind/models/lite-model_efficientdet_lite0_detection_metadata_1.tflite"
    LABEL_PATH = "/home/pi/mind/labelmap.txt"
    THRESHOLD = 0.25  # Порог уверенности

    # Инициализация детектора
    detector = ShipDetector(MODEL_PATH, LABEL_PATH, THRESHOLD)

    # Захват видео с камеры (для обработки изображения замените на cv2.imread)
    cap = cv2.VideoCapture("/home/pi/mind/videos/mil-test-low.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Детекция кораблей
        detections = detector.process(frame)
        print(detections)
        # Отрисовка результатов
        if detections:
            for (x1, y1, x2, y2, score) in detections:
                # Прямоугольник вокруг корабля
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Текст с уверенностью
                label = f"Ship: {score:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        # Показ кадра
        cv2.imshow('Ship Detection', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
