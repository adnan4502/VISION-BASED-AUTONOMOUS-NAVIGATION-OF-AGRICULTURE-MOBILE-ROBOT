from json import detect_encoding
from cv2 import detail_ProjectorBase
import jetson.inference
import jetson.utils
import time
import cv2
import numpy as np 
import RPi.GPIO as GPIO
import time
#!/usr/bin/env python
print("Successfully installed all libraries")

# for 1st Motor on ENA
ENA = 33
IN1 = 35
IN2 = 36
# FOR 2ND MOTOR ON ENB
ENB = 32
IN3 = 37
IN4 = 38

# set pin numbers to the board's
GPIO.setmode(GPIO.BOARD)

# initialize EnA, In1 and In2
GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)

GPIO.setup(ENB, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

# Stop
GPIO.output(ENA, GPIO.HIGH)
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(ENB, GPIO.HIGH)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)
time.sleep(1)



# # Stop
# GPIO.output(IN1, GPIO.LOW)
# GPIO.output(IN2, GPIO.LOW)
# GPIO.output(IN3, GPIO.LOW)
# GPIO.output(IN4, GPIO.LOW)
# time.sleep(1)

# # Backward
# GPIO.output(IN1, GPIO.LOW)
# GPIO.output(IN2, GPIO.HIGH)
# GPIO.output(IN3, GPIO.LOW)
# GPIO.output(IN4, GPIO.HIGH)
# time.sleep(1)

def distance_between_bounding_boxes(box1, box2):
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    return int(((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5)

timeStamp=time.time()
fpsFilt=0
net=jetson.inference.detectNet(argv=['--model=/home/nano/jetson-inference/python/training/detection/ssd/models/model0110/ssd-mobilenet.onnx',
                                    '--labels=/home/nano/jetson-inference/python/training/detection/ssd/models/model0110/labels.txt',
                                    '--input_blob=input_0', '--output_cvg=scores', '--output_bbox=boxes'],threshold=0.6)

dispW=800
dispH=600
flip=2
font=cv2.FONT_HERSHEY_SIMPLEX

# Gstreamer code for improvded Raspberry Pi Camera Quality
#camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'
#cam=cv2.VideoCapture(camSet)
#cam=jetson.utils.gstCamera(dispW,dispH,'0')
input_video = "test.mp4"
cam=cv2.VideoCapture(input_video)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)
print("Input video loaded")
#cam=jetson.utils.gstCamera(dispW,dispH,'/dev/video1')
#display=jetson.utils.glDisplay()
#while display.IsOpen():
while True:
    #img, width, height= cam.CaptureRGBA()
    _,img = cam.read()
    height=img.shape[0]
    width=img.shape[1]

    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    frame=jetson.utils.cudaFromNumpy(frame)

    detections=net.Detect(frame, width, height)
    for detect in detections:
        #print(detect)
        print("detect")
        ID=detect.ClassID
        item=net.GetClassDesc(ID)
        #if item='plant'
        top=int(detect.Top)
        left=int(detect.Left)
        bottom=int(detect.Bottom)
        right=int(detect.Right)
        width = int(detect.Width)
        height = int(detect.Height)

        box1 = (left, top, left + width, top + height)
        item=net.GetClassDesc(ID)
        print(item,top,left,bottom,right)
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
        cv2.putText(img,item,(left,top+20),font,.75,(0,0,255),2)

        # algorithm 1 starts for calculating distance between 2 bounding boxes
        previous_distance = None 
        for i in range(len(detections)):
            left1 = detections[i].Left
            top1 = detections[i].Top
            width1 = detections[i].Width
            height1 = detections[i].Height

            box1 = (left1, top1, left1 + width1, top1 + height1)
            if i == 0:
                left2 = detections[i + 1].Left
                top2 = detections[i + 1].Top
                width2 = detections[i + 1].Width
                height2 = detections[i + 1].Height

                box2 = (left2, top2, left2 + width2, top2 + height2)

                distance = distance_between_bounding_boxes(box1, box2)
                print("Distance: ", distance)
                if previous_distance != distance:
                    previous_distance = distance
                    print("Distance between bounding box", i+1, "and bounding box", i+2, "is: ", distance)
                    if distance>=80:
                        # Forward
                        print("Forward")
                        GPIO.output(IN1, GPIO.HIGH)
                        GPIO.output(IN2, GPIO.LOW)
                        GPIO.output(IN3, GPIO.HIGH)
                        GPIO.output(IN4, GPIO.LOW)
                    else:
                        # Stop
                        print("Stop")
                        GPIO.output(ENA, GPIO.LOW)
                        GPIO.output(IN1, GPIO.LOW)
                        GPIO.output(IN2, GPIO.LOW)
                        GPIO.output(ENB, GPIO.LOW)
                        GPIO.output(IN3, GPIO.LOW)
                        GPIO.output(IN4, GPIO.LOW)

        
        #algorithm 1 ends for calculating distance between 2 bounding boxes
    #display.RenderOnce(img,width,height)
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    fpsFilt=.9*fpsFilt + .1*fps
    #print(str(round(fps,1))+' fps')
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
GPIO.cleanup()
cv2.destroyAllWindows()
