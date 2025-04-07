import cv2
import time
import detection as dtc

video_path = 0
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("temp", frame)
    
    result = dtc.objection(frame)  # Вызываем функцию один раз
    
    if result == 1:
        print("Have any boat in frame")
    else:
        print("Victim")
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    time.sleep(0.01)
    
cap.release()
cv2.destroyAllWindows()
