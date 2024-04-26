from cv2 import *
import cv2
import os
import time
import requests
import numpy as np
from PIL import Image
from skimage.io import imread, imshow


cam = cv2.VideoCapture(0)       
time.sleep(5)
harcascadePath = "haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(harcascadePath)
sampleNum = 0
sid1= '240'
        
while True:
        ret, img = cam.read()
            
               #return "Failed to capture frame from webcam."            

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)            
        for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum +=1
                cv2.imwrite(f"TrainingImages/{sampleNum}.{sid1}.jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)                
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 30:
                break
        
cam.release()
cv2.destroyAllWindows()

