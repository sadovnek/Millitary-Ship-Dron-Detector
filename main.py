import cv2
import time
import detection as dtc

video_path = 0
cap = cv2.VideoCapture(video_path)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("H :" + str(height) + " W:" +str(width))
ret, frame = cap.read()

while True:
	ret, frame = cap.read()
	if not ret:
		break
	cv2.imshow("temp",frame)
	
	if(dtc.objection(frame)):
		print("Have any boat in frame")
		
	elif(dtc.objection(frame) == 0):
		print("Victim")
		
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break
	time.sleep(0.01)
	
cap.release()
cv2.destroyAllWindows()
