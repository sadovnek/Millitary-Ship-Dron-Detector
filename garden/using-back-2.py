import cv2
import time
import threading
from slib import YOLODetector

detector = YOLODetector('/home/pi/garden/best-l-fp16.tflite','/home/pi/garden/label.txt',0.4,0.7)
#detector = YOLODetector("/home/pi/tensorflow/TF-Lite-Python-Object-Objection/lite-model_yolo-v5-tflite_tflite_model_1.tflite","/home/pi/tensorflow/TF-Lite-Python-Object-Objection/labelmap.txt",0.2,0.5)
results = []
cord = []
frame = -1
cnt = 0
ret = -1
tracker = cv2.TrackerCSRT_create()
cap = cv2.VideoCapture(0)
state = 0
##bbox = (279, 338, 81, 92)

def find():
    global results, state, cnt
    time.sleep(0.5)
    while True:
        if state == 0:
            cv2.imwrite(f'./images/result_yolo_{cnt}.jpg', frame)
            results = detector.process(frame)
            print("NOW WAR")
            #scnt+=1
        elif state == 1:
            break
            #state = 0

threading.Thread(target=find, daemon=True).start()
score = 0
ret,frame = cap.read()

while True:
    #print("working")
    ret, frame = cap.read()
    FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
    pad = round(abs(FRAME_WIDTH - FRAME_HEIGHT) / 2)
    x_pad = pad if FRAME_HEIGHT > FRAME_WIDTH else 0
    y_pad = pad if FRAME_WIDTH > FRAME_HEIGHT else 0
    frame = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    #print(results)
    if results != [] and state == 0:
        for i in results:
            cv2.rectangle(frame, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,255,0), 2)
            coords = [i[0],i[1],i[2],i[3]]
        ret = tracker.init(frame, coords)
        state = 1
        
    elif state == 1:
        ret, coords = tracker.update(frame)
        p1 = (max(0,int(coords[0]-100)) , int(coords[1]))
        #p1 = (max(0,int(coords[0])) , int(coords[1]))
        #p2 = (int(coords[0] + coords[2]), int(coords[1] + coords[3]))
        p2 = (int(coords[0] + coords[2])+75, int(coords[1] + coords[3])+25)
        #print(p1[0], p1[1])
        label = f'Radar(FAR)\nHelicopter\nGUN: {score*100:.2f}%'
        cv2.putText(frame, "Radar(FAR)", (coords[0], coords[1]-55), cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
        cv2.putText(frame, "Helicopter", (coords[0], coords[1]-25), cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
        cv2.putText(frame, "GUN       ", (coords[0], coords[1]-40), cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
        cv2.putText(frame, "Missiles  ", (coords[0], coords[1]-10), cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
        cv2.rectangle(frame, p1, p2, (0,0,255), 2)
        #print(coords)
        cv2.imwrite(f'./images/result_yolo_{cnt}.jpg', frame)
    cv2.imshow('Object detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.05)
    
cap.release()
cv2.destroyAllWindows()

