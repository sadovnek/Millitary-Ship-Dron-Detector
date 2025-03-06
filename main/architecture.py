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
        time.sleep(1)
        if state == 0:
            cnt = random.randint(0,100)
            if cnt < 50:
                state = 1



threading.Thread(target=detect, daemon=True).start()
threading.Thread(target=type, daemon=True).start()
threading.Thread(target=track, daemon=True).start()

'''main'''
while(True):
    if(state == 2):
        print(f'x: {x} y: {y}')
    else:
        print(workings[state])
    time.sleep(0.5)
