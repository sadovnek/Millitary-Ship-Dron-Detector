"""HE MAIN PART"""

from slib import YOLODetector
import threading
import sys
import cv2
import time

results = []
coords = []
temp = []
tup = ()
arr = type(coords)
ret = False
state = 0
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
detector = YOLODetector("/home/pi/tensorflow/TF-Lite-Python-Object-Objection/lite-model_yolo-v5-tflite_tflite_model_1.tflite","/home/pi/tensorflow/TF-Lite-Python-Object-Objection/labelmap.txt",0.2,0.5)#7,3 Mib
creator = YOLODetector('/home/pi/garden/best-l-fp16.tflite','/home/pi/garden/label.txt',0.4,0.65)#13.5 Mib

def have():
	global results
	time.sleep(0.5)
	while True:
		print(f'state in have:{state}')
		if state == 0:
			print("THREaDING HAVE", f'state:{state}')
			results = detector.process(frame)
		else:
			print("I am die - HAVE",f'state:{state}')
			break
			print("I am alive - HAVE")

def war():
	global coords
	time.sleep(0.5)
	while True:
		print(f'state in war:{state}')
		if state == 0:
			print("THREaDING WAR", f'state:{state}')
			coords = creator.process(frame)
		else:
			print("I am die - WAR",f'state:{state}')
			break
			print("I am alive - WAR")

threading.Thread(target=war, daemon=True).start()
threading.Thread(target=have, daemon=True).start()

while True:
	ret, frame = cap.read()
	FRAME_HEIGHT, FRAME_WIDTH = frame.shape[:2]
	pad = round(abs(FRAME_WIDTH - FRAME_HEIGHT) / 2)
	x_pad = pad if FRAME_HEIGHT > FRAME_WIDTH else 0
	y_pad = pad if FRAME_WIDTH > FRAME_HEIGHT else 0
	frame = cv2.copyMakeBorder(frame, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
	if results != []:
		for ship in results:
			if ship[5] == "ship" and state == 0:
				cv2.putText(frame, ship[5]+ ' '+str(round(ship[4],2)), (ship[0], ship[1]-10), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
				cv2.rectangle(frame, (ship[0],ship[1]), (ship[0]+ship[2],ship[1]+ship[3]), (0,255,0), 2)
				
	if coords != []:
		for i in coords:
			if state == 0:
				cv2.putText(frame, i[5]+ ' '+str(i[4]), (i[0], i[1]-10), cv2.FONT_ITALIC, 0.5, (255,255,55), 1, 0)
				cv2.rectangle(frame, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (0,255,0), 2)
				#coords = [i[0]+10,i[1],i[2],i[3]]
				temp = [i[0]+10,i[1],i[2],i[3]]
				ret = tracker.init(frame, temp)
				state = 1
	
	if state == 1:
		#print("NO DIFS OF COORDS", f"state: {state}")
		time.sleep(0.01)
		ret, temp = tracker.update(frame)
		#print(coords)
		#print(type(coords))
		#print(len(tup))
		if temp == [] or len(temp) == 0 or type(temp) == arr:#list index out of range
			continue
		else:
			print(temp[0],temp[1],temp[2],temp[3])
			p1 = (max(0,int(temp[0]- 150)) , max(int(temp[1]),0))
			p2 = (max(0,int(temp[0] + temp[2])+105), max(int(temp[1] + temp[3]+25),0))
			
			cv2.putText(frame, "Radar(FAR)", (max(int(temp[0]),0), max(int(temp[1]-55),0)), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
			cv2.putText(frame, "Helicopter", (max(int(temp[0]),0), max(int(temp[1]-25),0)), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
			cv2.putText(frame, "GUN       ", (max(int(temp[0]),0), max(int(temp[1]-40),0)), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
			cv2.putText(frame, "Missiles  ", (max(int(temp[0]),0), max(int(temp[1]-10),0)), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
			cv2.rectangle(frame, p1, p2, (0,0,255), 2)
		
	cv2.imshow('Object detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	if state == 0:
		time.sleep(0.05) 
    
cap.release()
cv2.destroyAllWindows()
