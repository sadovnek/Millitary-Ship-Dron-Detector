#import cv2
#import torch
#import os
import cv2
import time
import random
import threading

workings = ["detect","type","tracking"]
state = 0
x = 0; y =0
frame = -1
ret = None

def track():
    global state
    global x,y
    while(True):
        time.sleep(1)
        if(state == 2):
            x = random.randint(0,100)
            y = random.randint(0,100)
            

def type():
    global state
    while(True):
        time.sleep(1)
        if(state == 1):
            dif = random.randint(0,100)
            if dif < 25:
                state = 2
    
def detect():
    global state
    while(True):
        if not ret:
            print("Error: Could not read frame.")
        else:
            print("All good!")
        #time.sleep(0.001)
        
        '''time.sleep(1)
        if state == 0:
            cnt = random.randint(0,100)
            if cnt < 50:
                state = 1'''



threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=type, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

'''main'''
video_path = 'large.mp4'
cap = cv2.VideoCapture(video_path)


while(True):
    time.sleep(5)
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cv2.imshow('Boat Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
