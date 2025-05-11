"""HE MAIN PART"""

from slib import YOLODetector
import threading
import cv2
import time
import traceback

results = []
coords = []
ret = False
state = 0
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
detector = YOLODetector("/home/pi/tensorflow/TF-Lite-Python-Object-Objection/lite-model_yolo-v5-tflite_tflite_model_1.tflite","/home/pi/tensorflow/TF-Lite-Python-Object-Objection/labelmap.txt",0.2,0.5)
creator = YOLODetector('/home/pi/garden/best-l-fp16.tflite','/home/pi/garden/label.txt',0.3,0.65)

def have():
    global results
    time.sleep(1.5)
    while True:
        try:
            if state == 0:
                results = detector.process(frame)
            else:
                break
        except Exception:
            print(f"Error in detection thread: {traceback.format_exc()}")
            time.sleep(0.1)

def war():
    global coords
    time.sleep(1.5)
    while True:
        try:
            if state == 0:
                coords = creator.process(frame)
            else:
                break
        except Exception:
            print(f"Error in coordinate thread: {traceback.format_exc()}")
            time.sleep(0.1)

threading.Thread(target=have, daemon=True).start()
threading.Thread(target=war, daemon=True).start()

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            time.sleep(0.1)
            continue
        
        FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
        pad = round(abs(FRAME_WIDTH - FRAME_HEIGHT) / 2)
        x_pad = pad if FRAME_HEIGHT > FRAME_WIDTH else 0
        y_pad = pad if FRAME_WIDTH > FRAME_HEIGHT else 0
        frame = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        try:
            if results:
                for ship in results:
                    # Check array length before accessing indexes
                    if len(ship) >= 6 and ship[5] == "ship" and state == 0:
                        try:
                            cv2.putText(frame, f"{ship[5]} {ship[4]}", (ship[0], ship[1]-10), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
                            cv2.rectangle(frame, (ship[0],ship[1]), (ship[0]+ship[2],ship[1]+ship[3]), (0,255,0), 2)
                        except IndexError:
                            print(f"Invalid ship coordinates: {ship}")
                            continue
        except Exception:
            print(f"Error drawing ship results: {traceback.format_exc()}")
            continue

        try:
            if coords:
                for i in coords:
                    # Check array length before accessing indexes
                    if len(i) >= 6 and state == 0:
                        try:
                            cv2.putText(frame, f"{i[5]} {i[4]}", (i[0], i[1]-10), cv2.FONT_ITALIC, 0.5, (255,255,55), 1, 0)
                            cv2.rectangle(frame, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (0,255,0), 2)
                            # Ensure coordinates are valid
                            new_coords = [
                                max(i[0]+10, 0),
                                max(i[1], 0),
                                max(i[2], 0),
                                max(i[3], 0)
                            ]
                            if all(isinstance(n, (int, float)) for n in new_coords):
                                coords = new_coords
                                ret = tracker.init(frame, coords)
                                state = 1
                        except IndexError:
                            print(f"Invalid coordinate array: {i}")
                            continue
        except Exception:
            print(f"Error processing coordinates: {traceback.format_exc()}")
            continue

        try:
            if state == 1:
                time.sleep(0.01)
                print(coords)
                ret, coords = tracker.update(frame)
                if coords and len(coords) >= 4:
                    try:
                        p1 = (max(0,int(coords[0]-150)), max(int(coords[1]),0))
                        p2 = (max(0,int(coords[0] + coords[2])+105), max(int(coords[1] + coords[3]+25),0))
                        # Add text drawing with checks
                        texts = [
                            ("Radar(FAR)", -55),
                            ("Helicopter", -25),
                            ("GUN", -40),
                            ("Missiles", -10)
                        ]
                        for text, offset in texts:
                            y_pos = max(int(coords[1] + offset), 0)
                            cv2.putText(frame, text, (max(int(coords[0]),0), y_pos), cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
                        cv2.rectangle(frame, p1, p2, (0,0,255), 2)
                    except IndexError:
                        print(f"Invalid tracking coordinates: {coords}")
                        state = 0
        except Exception:
            print(f"Error in tracking state: {traceback.format_exc()}")
            state = 0
            continue

        cv2.imshow('Object detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
        
    except Exception:
        print(f"Critical error in main loop: {traceback.format_exc()}")
        time.sleep(0.5)
        continue

cap.release()
cv2.destroyAllWindows()
