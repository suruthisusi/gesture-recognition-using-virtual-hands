import cv2
from cfg import cfg
import mediapipe as mp
from drawpage import DrawPage
from detectclick import DetectClick, Click, Amt
import datetime
from cv2 import *
import os
import time
import requests
import numpy as np
from PIL import Image
from skimage.io import imread, imshow
import tkinter.font as font
from time import sleep
from pynput.keyboard import Controller, Key
from scipy.spatial import distance as dist
import threading
import cv2
import cvzone
import dlib
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector


name={240:{"password":3005,"Name":"Kalai","Acc_No":1928374650,"Balance": 5000}}

def getImagesAndLabels(path):
        imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
        faces=[]
        Ids=[]
        for imagePath in imagePaths:
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            Id=int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        return faces,Ids

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()  
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Id))
    recognizer.save("train.yml")
 
TrainImages()

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]),
                      (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out
class MyButton():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

recognizer = cv2.face_LBPHFaceRecognizer.create()
recognizer.read("train.yml")
harcascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(harcascadePath)

def main():
    cap = cv2.VideoCapture(0)    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
    dwPg = DrawPage(cfg["pages"])
    coords = dwPg.getCoordintates()

    detClick = DetectClick([s[-1] for s in coords])

    mp_hands = mp.solutions.hands.Hands(
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
        max_num_hands=cfg["max_num_hands"]
    )

    with mp_hands as hands:
        while cap.isOpened():
            ret, im = cap.read()

            if not ret:
                print("Ignoring empty camera frame.")
                break

            overlay = cv2.flip(im, 1).copy()
            output = cv2.flip(im, 1).copy()   
           
            
            if cfg["currentpage"] == "Match":
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                faces = faceCascade.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    Id, cfg["face_matched"] = recognizer.predict(gray[y:y + h, x:x + w]) 
                    
                    if cfg["face_matched"] < 50:                         
                                    Id=str(Id)
                                    print(Id)
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    cap_ = cv2.VideoCapture(0)
                                    cap_.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                    cap_.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                            #ret, img = cap_.read()
                                    detector = HandDetector(detectionCon=1)
                                    keys = [["7", "8", "9"],
                                            ["4", "5", "6"],
                                            ["1", "2", "3"],
                                            ["0", ".", "<-"]]

                                    finalText = ""
                                    keyboard = Controller()

                                    buttonList = []
                                    for i in range(len(keys)):
                                                for j, key in enumerate(keys[i]):
                                                    buttonList.append(MyButton([100 * j + 50, 100 * i + 150], key))


                                    while True:
                                                success, img = cap_.read()
                                                im = detector.findHands(img)
                                                lmList, bboxInfo = detector.findPosition(im)
                                                im = drawAll(im, buttonList)

                                                if not success:
                                                    print("Error reading frame from the camera.")
                                                    break
                                               
                                                if len(finalText) == 4:
                                                        if int(finalText) == name[int(Id)]["password"]:
                                                                cap_.release()
                                                                cv2.destroyAllWindows()
                                                                cap3 = cv2.VideoCapture(0)
                                                                cap3.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                                                cap3.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                                                dwPg = DrawPage(cfg["pages"])
                                                                coords = dwPg.getCoordintates()

                                                                detClicks = Click([s[-1] for s in coords])

                                                                mp_hands = mp.solutions.hands.Hands(
                                                                        min_detection_confidence=cfg["min_detection_confidence"],
                                                                        min_tracking_confidence=cfg["min_tracking_confidence"],
                                                                        max_num_hands=cfg["max_num_hands"]
                                                                    )
                                                                balance = ""

                                                                with mp_hands as hands:
                                                                        while cap3.isOpened():
                                                                            ret, im = cap3.read()

                                                                            if not ret:
                                                                                print("Ignoring empty camera frame.")
                                                                                break

                                                                            overlay = cv2.flip(im, 1).copy()
                                                                            output = cv2.flip(im, 1).copy()
                                                                            
                                                                            if cfg["curretpage"] == "Balance":
                                                                                    if int(Id) in name and "Balance" in name[int(Id)]:
                                                                                        balance = name[int(Id)]["Balance"]
                                                                                        
                                                                                        cfg["pages"]["balan"] = {
                                                                                                "pagetitle": [f"Select Balance is {balance}", 100, 1.8, (175, 0, 175), 4],
                                                                                                "buttons": ["", "", "", "", "Back", "", "", ""],
                                                                                                "navigation": ["", "", "", "", "Transactions", "", "", ""]
                                                                                            }
                                                                                        
                                                                                        

                                                                                        cfg["curretpage"] = "balan"
                                                                            if cfg["curretpage"] == "Exit":
                                                                                    exit()

                                                                            if cfg["curretpage"] == "Deposit":
                                                                                    cap3.release()
                                                                                    cap6 = cv2.VideoCapture(0)
                                                                                    cap6.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                                                                    cap6.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                                                                    while cap6.isOpened():
                                                                                            ret, im = cap6.read()
                                                                                            detector = HandDetector(detectionCon=1)
                                                                                            keys = [["7", "8", "9", "s"],
                                                                                                    ["4", "5", "6"],
                                                                                                    ["1", "2", "3"],
                                                                                                    ["0", ".", "<-"]
                                                                                                    ]

                                                                                            amount = ""

                                                                                            keyboard = Controller()

                                                                                            buttonList = []
                                                                                            for i in range(len(keys)):
                                                                                                for j, key in enumerate(keys[i]):
                                                                                                    buttonList.append(MyButton([100 * j + 50, 100 * i + 150], key))


                                                                                            while True:
                                                                                                success, im = cap6.read()
                                                                                                im = detector.findHands(im)
                                                                                                lmList, bboxInfo = detector.findPosition(im)
                                                                                                im = drawAll(im, buttonList)
                                                                                                if amount and amount[-1] == 's':
                                                                                                        amount_without_s = amount[:-1]
                                                                                                        if  amount_without_s:
                                                                                                                cap6.release()
                                                                                                                cv2.destroyAllWindows()
                                                                                                                cap7 = cv2.VideoCapture(0)
                                                                                                                cap7.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                                                                                                cap7.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                                                                                                dwPg = DrawPage(cfg["pages"])
                                                                                                                coords = dwPg.getCoordintates()

                                                                                                                detClicks = Amt([s[-1] for s in coords])

                                                                                                                mp_hands = mp.solutions.hands.Hands(
                                                                                                                        min_detection_confidence=cfg["min_detection_confidence"],
                                                                                                                        min_tracking_confidence=cfg["min_tracking_confidence"],
                                                                                                                        max_num_hands=cfg["max_num_hands"]
                                                                                                                    )
                                                                                                                balance = ""

                                                                                                                with mp_hands as hands:
                                                                                                                        while cap7.isOpened():
                                                                                                                            ret, im = cap7.read()

                                                                                                                            if not ret:
                                                                                                                                print("Ignoring empty camera frame.")
                                                                                                                                break

                                                                                                                            overlay = cv2.flip(im, 1).copy()
                                                                                                                            output = cv2.flip(im, 1).copy()

                                                                                                                            if cfg["curpage"] == "Balance":
                                                                                                                                    if int(Id) in name and "Balance" in name[int(Id)]:
                                                                                                                                        amount_without_s = amount[:-1]
                                                                                                                                        balance = name[int(Id)]["Balance"]
                                                                                                                                        value = balance + int(amount_without_s)

                                                                                                                                        
                                                                                                                                        cfg["pages"]["balan"] = {
                                                                                                                                                "pagetitle": [f"Select Balance is {value}", 100, 1.8, (175, 0, 175), 4],
                                                                                                                                                "buttons": ["", "", "", "", "Back", "", "", ""],
                                                                                                                                                "navigation": ["", "", "", "", "Transactions", "", "", ""]
                                                                                                                                            }
                                                                                                                                        cfg["curpage"] = "balan"
                                                                                                                            if cfg["curpage"] == "Exit":
                                                                                                                                    exit()
                                                                                                                                                                       

                                                                                                                            dwPg.drawThePage(cfg["curpage"], overlay)                                                                          

                                                                           

                                                                                                                            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                                                                                                                            output.flags.writeable = False
                                                                                                                        
                                                                                                                            results = hands.process(output)

                                                                                                                            output.flags.writeable = True
                                                                                                                            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                                                                                                                            if results and results.multi_hand_landmarks:
                                                                                                                                        hand = results.multi_hand_landmarks[0]

                                                                                                                                        imgH, imgW, imgC = output.shape

                                                                                                                                        xpos, ypos = int(hand.landmark[8].x * imgW), int(hand.landmark[8].y * imgH)

                                                                                                                                        cv2.circle(overlay, (xpos, ypos), 20, (255, 0, 255), cv2.FILLED)

                                                                                                                                        clickedBtnIndex = detClicks.detectkey((xpos, ypos))

                                                                                                                                        if clickedBtnIndex is not None and clickedBtnIndex > -1 and clickedBtnIndex < 8:

                                                                                                                                            if "endtime" not in cfg:
                                                                                                                                                cfg["endtime"] = datetime.datetime.now() + datetime.timedelta(seconds=cfg["btnClickDelay"])
                                                                                                                                                cfg["endtime"] = cfg["endtime"].strftime("%H:%M:%S.%f")[:-5]

                                                                                                                                            elif cfg["endtime"] <= datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]:
                                                                                                                                                cPage = cfg["curpage"]
                                                                                                                                                cfg["curpage"] = cfg["pages"][cPage]["navigation"][clickedBtnIndex]
                                                                                                                                                del cfg["endtime"]

                                                                                                                                            elif clickedBtnIndex is None and "endtime" in cfg:
                                                                                                                                                        if "face_matched" in cfg:
                                                                                                                                                                del cfg["face_matched"]                                        
                                                ##
                                                                                                                            alpha = cfg["alpha"]
                                                                                                                            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                                                                                                                            balance = f'Balance: {balance}'
                                                                                                                            cv2.putText(im, balance, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                                                                                                                            cv2.imshow("ATM", output)

                                                                                                                            if cv2.waitKey(5) & 0xFF == 27 or cfg["curpage"] == "Exit":
                                                                                                                                break

                                                                                                                #cap5.release()
                                                                                                                #cv2.destroyAllWindows()        
                                                                                                            
                                                                                                        else:
                                                                                                                cap6.release()
                                                                                                                cv2.destroyAllWindows()
                                                                                                                print('Invalid amount length')
                                                                                                                break
                                                                                                if lmList:
                                                                                                    for button in buttonList:
                                                                                                        x, y = button.pos
                                                                                                        w, h = button.size

                                                                                                        if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                                                                                                            cv2.rectangle(im, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                                                                                                            cv2.putText(im, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

                                                                                                            l, _, _ = detector.findDistance(8, 12, im, draw=False)
                                                                                                            
                                                                                                            if l < 10:
                                                                                                                if button.text == "<-":
                                                                                                                    if len(finalText):
                                                                                                                        keyboard.press(Key.backspace)
                                                                                                                        finalText = finalText.strip(finalText[-1])
                                                                                                                        cv2.rectangle(im, button.pos, (x + w, y + h),
                                                                                                                                      (0, 255, 0), cv2.FILLED)
                                                                                                                        cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                                                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)                                                                
                                                                                                                else:
                                                                                                                    keyboard.press(button.text)
                                                                                                                    cv2.rectangle(im, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                                                                                                    cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                                                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                                                                                                    amount += button.text



                                                                                                                    sleep(1.0)

                                                                                                cv2.rectangle(im, (30, 2), (1160, 120), (175, 0, 175), cv2.FILLED)
                                                                                               
                                                                                                balances = f'Enter Amount: {amount}'
                                                                                                cv2.putText(im, balances, (40, 120),
                                                                                                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

                                                                                                cv2.imshow("ATM", im)
                                                                                                if cv2.waitKey(5) & 0xFF == 27 or cfg["curpage"] == "Exit":
                                                                                                    break
                                                                                    

                                                                            
                                                                            if cfg["curretpage"] == "Withdraw":
                                                                                    cap3.release()
                                                                                    cap4 = cv2.VideoCapture(0)
                                                                                    cap4.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                                                                    cap4.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                                                                    while cap4.isOpened():
                                                                                            ret, im = cap4.read()
                                                                                            detector = HandDetector(detectionCon=1)
                                                                                            keys = [["7", "8", "9", "s"],
                                                                                                    ["4", "5", "6"],
                                                                                                    ["1", "2", "3"],
                                                                                                    ["0", ".", "<-"]
                                                                                                    ]

                                                                                            amount = ""

                                                                                            keyboard = Controller()

                                                                                            buttonList = []
                                                                                            for i in range(len(keys)):
                                                                                                for j, key in enumerate(keys[i]):
                                                                                                    buttonList.append(MyButton([100 * j + 50, 100 * i + 150], key))


                                                                                            while True:
                                                                                                success, im = cap4.read()
                                                                                                im = detector.findHands(im)
                                                                                                lmList, bboxInfo = detector.findPosition(im)
                                                                                                im = drawAll(im, buttonList)
                                                                                                if amount and amount[-1] == 's':
                                                                                                        amount_without_s = amount[:-1]
                                                                                                        if  5000 >= int(amount_without_s) >= 100:
                                                                                                                cap4.release()
                                                                                                                cv2.destroyAllWindows()
                                                                                                                cap5 = cv2.VideoCapture(0)
                                                                                                                cap5.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["screen_x"])
                                                                                                                cap5.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["screen_y"])
                                                                                                                dwPg = DrawPage(cfg["pages"])
                                                                                                                coords = dwPg.getCoordintates()

                                                                                                                detClicks = Amt([s[-1] for s in coords])

                                                                                                                mp_hands = mp.solutions.hands.Hands(
                                                                                                                        min_detection_confidence=cfg["min_detection_confidence"],
                                                                                                                        min_tracking_confidence=cfg["min_tracking_confidence"],
                                                                                                                        max_num_hands=cfg["max_num_hands"]
                                                                                                                    )
                                                                                                                balance = ""

                                                                                                                with mp_hands as hands:
                                                                                                                        while cap5.isOpened():
                                                                                                                            ret, im = cap5.read()

                                                                                                                            if not ret:
                                                                                                                                print("Ignoring empty camera frame.")
                                                                                                                                break

                                                                                                                            overlay = cv2.flip(im, 1).copy()
                                                                                                                            output = cv2.flip(im, 1).copy()

                                                                                                                            if cfg["curpage"] == "Balance":
                                                                                                                                    if int(Id) in name and "Balance" in name[int(Id)]:
                                                                                                                                        amount_without_s = amount[:-1]
                                                                                                                                        balance = name[int(Id)]["Balance"]
                                                                                                                                        value = balance - int(amount_without_s)

                                                                                                                                        
                                                                                                                                        cfg["pages"]["balan"] = {
                                                                                                                                                "pagetitle": [f"Select Balance is {value}", 100, 1.8, (175, 0, 175), 4],
                                                                                                                                                "buttons": ["", "", "", "", "Back", "", "", ""],
                                                                                                                                                "navigation": ["", "", "", "", "Transactions", "", "", ""]
                                                                                                                                            }
                                                                                                                                        cfg["curpage"] = "balan"
                                                                                                                            if cfg["curpage"] == "Exit":
                                                                                                                                    exit()
                                                                                                                                                                       

                                                                                                                            dwPg.drawThePage(cfg["curpage"], overlay)                                                                          

                                                                           

                                                                                                                            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                                                                                                                            output.flags.writeable = False
                                                                                                                        
                                                                                                                            results = hands.process(output)

                                                                                                                            output.flags.writeable = True
                                                                                                                            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                                                                                                                            if results and results.multi_hand_landmarks:
                                                                                                                                        hand = results.multi_hand_landmarks[0]

                                                                                                                                        imgH, imgW, imgC = output.shape

                                                                                                                                        xpos, ypos = int(hand.landmark[8].x * imgW), int(hand.landmark[8].y * imgH)

                                                                                                                                        cv2.circle(overlay, (xpos, ypos), 20, (255, 0, 255), cv2.FILLED)

                                                                                                                                        clickedBtnIndex = detClicks.detectkey((xpos, ypos))

                                                                                                                                        if clickedBtnIndex is not None and clickedBtnIndex > -1 and clickedBtnIndex < 8:

                                                                                                                                            if "endtime" not in cfg:
                                                                                                                                                cfg["endtime"] = datetime.datetime.now() + datetime.timedelta(seconds=cfg["btnClickDelay"])
                                                                                                                                                cfg["endtime"] = cfg["endtime"].strftime("%H:%M:%S.%f")[:-5]

                                                                                                                                            elif cfg["endtime"] <= datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]:
                                                                                                                                                cPage = cfg["curpage"]
                                                                                                                                                cfg["curpage"] = cfg["pages"][cPage]["navigation"][clickedBtnIndex]
                                                                                                                                                del cfg["endtime"]

                                                                                                                                            elif clickedBtnIndex is None and "endtime" in cfg:
                                                                                                                                                        if "face_matched" in cfg:
                                                                                                                                                                del cfg["face_matched"]                                        
                                                ##
                                                                                                                            alpha = cfg["alpha"]
                                                                                                                            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                                                                                                                            balance = f'Balance: {balance}'
                                                                                                                            cv2.putText(im, balance, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                                                                                                                            cv2.imshow("ATM", output)

                                                                                                                            if cv2.waitKey(5) & 0xFF == 27 or cfg["curpage"] == "Exit":
                                                                                                                                break

                                                                                                                #cap5.release()
                                                                                                                #cv2.destroyAllWindows()        
                                                                                                            
                                                                                                        else:
                                                                                                                cap4.release()
                                                                                                                cv2.destroyAllWindows()
                                                                                                                print('Invalid amount length')
                                                                                                                break
                                                                                                if lmList:
                                                                                                    for button in buttonList:
                                                                                                        x, y = button.pos
                                                                                                        w, h = button.size

                                                                                                        if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                                                                                                            cv2.rectangle(im, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                                                                                                            cv2.putText(im, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

                                                                                                            l, _, _ = detector.findDistance(8, 12, im, draw=False)
                                                                                                            
                                                                                                            if l < 10:
                                                                                                                if button.text == "<-":
                                                                                                                    if len(finalText):
                                                                                                                        keyboard.press(Key.backspace)
                                                                                                                        finalText = finalText.strip(finalText[-1])
                                                                                                                        cv2.rectangle(im, button.pos, (x + w, y + h),
                                                                                                                                      (0, 255, 0), cv2.FILLED)
                                                                                                                        cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                                                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)                                                                
                                                                                                                else:
                                                                                                                    keyboard.press(button.text)
                                                                                                                    cv2.rectangle(im, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                                                                                                    cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                                                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                                                                                                    amount += button.text



                                                                                                                    sleep(1.0)

                                                                                                cv2.rectangle(im, (30, 2), (1160, 120), (175, 0, 175), cv2.FILLED)
                                                                                               
                                                                                                balances = f'Enter Amount: {amount}'
                                                                                                cv2.putText(im, balances, (40, 120),
                                                                                                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

                                                                                                cv2.imshow("ATM", im)
                                                                                                if cv2.waitKey(5) & 0xFF == 27 or cfg["curpage"] == "Exit":
                                                                                                    break

                                                                                      
                                                                                                                                               
                                                                                                                                                                                                                                    
                                                                            
                                                                                    
                                                                                    
                                                                                    
                                                                                    

                                                                            
                                                                                    
                                                                            dwPg.drawThePage(cfg["curretpage"], overlay)                                                                          

                                                                           

                                                                            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                                                                            output.flags.writeable = False
                                                                        
                                                                            results = hands.process(output)

                                                                            output.flags.writeable = True
                                                                            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                                                                            if results and results.multi_hand_landmarks:
                                                                                        hand = results.multi_hand_landmarks[0]

                                                                                        imgH, imgW, imgC = output.shape

                                                                                        xpos, ypos = int(hand.landmark[8].x * imgW), int(hand.landmark[8].y * imgH)

                                                                                        cv2.circle(overlay, (xpos, ypos), 20, (255, 0, 255), cv2.FILLED)

                                                                                        clickedBtnIndex = detClicks.detect((xpos, ypos))

                                                                                        if clickedBtnIndex is not None and clickedBtnIndex > -1 and clickedBtnIndex < 8:

                                                                                            if "endtime" not in cfg:
                                                                                                cfg["endtime"] = datetime.datetime.now() + datetime.timedelta(seconds=cfg["btnClickDelay"])
                                                                                                cfg["endtime"] = cfg["endtime"].strftime("%H:%M:%S.%f")[:-5]

                                                                                            elif cfg["endtime"] <= datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]:
                                                                                                cPage = cfg["curretpage"]
                                                                                                cfg["curretpage"] = cfg["pages"][cPage]["navigation"][clickedBtnIndex]
                                                                                                del cfg["endtime"]

                                                                                            elif clickedBtnIndex is None and "endtime" in cfg:
                                                                                                        if "face_matched" in cfg:
                                                                                                                del cfg["face_matched"]                                        
##
                                                                            alpha = cfg["alpha"]
                                                                            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
                                                                            balance = f'Balance: {balance}'
                                                                            cv2.putText(im, balance, (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

                                                                            cv2.imshow("ATM", output)

                                                                            if cv2.waitKey(5) & 0xFF == 27 or cfg["curretpage"] == "Exit":                                                                                
                                                                                cv2.destroyAllWindows()

                                                                cap3.release()
                                                                cv2.destroyAllWindows()
                                                                                                                                                                         
                                                        else:
                                                                break
                                                                
                                                
                                                
                                                if lmList:
                                                    for button in buttonList:
                                                        x, y = button.pos
                                                        w, h = button.size

                                                        if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                                                            cv2.rectangle(im, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                                                            cv2.putText(im, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 4)

                                                            l, _, _ = detector.findDistance(8, 12, im, draw=False)
                                                            
                                                            if l < 10:
                                                                if button.text == "<-":
                                                                    if len(finalText):
                                                                        keyboard.press(Key.backspace)
                                                                        finalText = finalText.strip(finalText[-1])
                                                                        cv2.rectangle(im, button.pos, (x + w, y + h),
                                                                                      (0, 255, 0), cv2.FILLED)
                                                                        cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)                                                                
                                                                else:
                                                                    keyboard.press(button.text)
                                                                    cv2.rectangle(im, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                                                    cv2.putText(im, button.text, (x + 20, y + 65),
                                                                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                                                                    finalText += button.text
                                                                    #print(finalText)



                                                                    sleep(1.0)

                                                cv2.rectangle(im, (90, 2), (1160, 120), (175, 0, 175), cv2.FILLED)
                                                Id_ = f'ID: {Id}'
                                                cv2.putText(im, Id_, (400, 50), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)
                                                masked_pin = '*' * len(finalText)

                                                final = f'PIN: {masked_pin}'
                                                cv2.putText(im, final, (400, 120),
                                                            cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

                                                cv2.imshow("ATM", im)
                                                if cv2.waitKey(1) & 0xFF == ord('q'):                                                        
                                                        break

                                                                        


                    else:
                            cfg["currentpage"] = "home"
                            continue

                  
                     
                
            dwPg.drawThePage(cfg["currentpage"], overlay)

            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output.flags.writeable = False
        
            results = hands.process(output)

            output.flags.writeable = True
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            if results and results.multi_hand_landmarks:
                        hand = results.multi_hand_landmarks[0]

                        imgH, imgW, imgC = output.shape

                        xpos, ypos = int(hand.landmark[8].x * imgW), int(hand.landmark[8].y * imgH)

                        cv2.circle(overlay, (xpos, ypos), 20, (255, 0, 255), cv2.FILLED)

                        clickedBtnIndex = detClick.detectClick((xpos, ypos))

                        if clickedBtnIndex is not None and clickedBtnIndex > -1 and clickedBtnIndex < 8:

                            if "endtime" not in cfg:
                                cfg["endtime"] = datetime.datetime.now() + datetime.timedelta(seconds=cfg["btnClickDelay"])
                                cfg["endtime"] = cfg["endtime"].strftime("%H:%M:%S.%f")[:-5]

                            elif cfg["endtime"] <= datetime.datetime.now().strftime("%H:%M:%S.%f")[:-5]:
                                cPage = cfg["currentpage"]
                                cfg["currentpage"] = cfg["pages"][cPage]["navigation"][clickedBtnIndex]
                                del cfg["endtime"]

                        elif clickedBtnIndex is None and "endtime" in cfg:
                                if "face_matched" in cfg:
                                        del cfg["face_matched"]                                        

            alpha = cfg["alpha"]
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            cv2.imshow("ATM", output)

            if cv2.waitKey(5) & 0xFF == 27 or cfg["currentpage"] == "Exit":
                break
            

    cap.release()
    cv2.destroyAllWindows()



main()
