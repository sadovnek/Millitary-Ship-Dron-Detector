"""HE MAIN PART"""

from slib import YOLODetector
import threading
import cv2
import time

results = []
coords = []
ret = False
state = 0
cap = cv2.VideoCapture(0)
tracker = cv2.TrackerCSRT_create()
detector = YOLODetector("/home/pi/tensorflow/TF-Lite-Python-Object-Objection/lite-model_yolo-v5-tflite_tflite_model_1.tflite","/home/pi/tensorflow/TF-Lite-Python-Object-Objection/labelmap.txt",0.2,0.5)
creator = YOLODetector('/home/pi/garden/best-l-fp16.tflite','/home/pi/garden/label.txt',0.2,0.65)

def have():
	global results
	time.sleep(0.5)
	while True:
		if state == 0:
			results = detector.process(frame)
		else:
			break

def war():
	global coords
	time.sleep(0.5)
	while True:
		if state == 0:
			coords = creator.process(frame)
		else:
			break

threading.Thread(target=have, daemon=True).start()
threading.Thread(target=war, daemon=True).start()

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
				cv2.putText(frame, ship[5]+ ' '+str(ship[4]), (ship[0], ship[1]-10), cv2.FONT_ITALIC, 0.5, (255,0,0), 1, 0)
				cv2.rectangle(frame, (ship[0],ship[1]), (ship[0]+ship[2],ship[1]+ship[3]), (0,255,0), 2)
				
	if coords == [] or len(coords) != 0:
		for i in coords:
			if state == 0:
				cv2.putText(frame, i[5]+ ' '+str(i[4]), (i[0], i[1]-10), cv2.FONT_ITALIC, 0.5, (255,255,55), 1, 0)
				cv2.rectangle(frame, (i[0],i[1]), (i[0]+i[2],i[1]+i[3]), (0,255,0), 2)
				coords = [i[0]+10,i[1],i[2],i[3]]
				ret = tracker.init(frame, coords)
				state = 1
	
	if state == 1:
		time.sleep(0.05)
		ret, coords = tracker.update(frame)
		if not ret:  # Check if tracker update was unsuccessful
			print("Tracker lost the target. Resetting...")
			state = 0
			coords = []
			continue
		try:
			# Ensure coordinates are valid and accessible
			x, y, w, h = map(int, coords)  # Convert to integers
			print(type(x), type(y), type(w), type(h))
			# Calculate positions with boundary checks
			p1 = (max(0, x - 150), max(y, 0))
			p2 = (max(0, x + w + 105), max(y + h + 25, 0))
			# Text positions adjusted to prevent negative values
			text_positions = [
				(x, y - 55),  # Radar
				(x, y - 40),  # GUN
				(x, y - 25),  # Helicopter
				(x, y - 10)    # Missiles
			]
			texts = ["Radar(FAR)", "GUN       ", "Helicopter", "Missiles  "]
			# Draw annotations
			for text, (tx, ty) in zip(texts, text_positions):
				cv2.putText(frame, text, (max(tx, 0), max(ty, 0)), 
					cv2.FONT_ITALIC, 0.5, (0,100,255), 1, 0)
			cv2.rectangle(frame, p1, p2, (0,0,255), 2)
			
		except (IndexError, ValueError) as e:
			print(f"Coordinate error: {e}. Resetting tracker.")
			state = 0
			coords = []
		except Exception as e:
			print(f"Unexpected error: {e}")
			state = 0
			coords = []
		
	cv2.imshow('Object detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	time.sleep(0.05)
    
cap.release()
cv2.destroyAllWindows()
