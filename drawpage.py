from cfg import cfg
from operator import itemgetter
from pynput.keyboard import Controller, Key
from scipy.spatial import distance as dist
import threading
import cv2
import cvzone
import dlib
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector
import numpy as np


class DrawPage():

    def __init__(self, pages):
        self.pages = pages
    def clearPage(self, img):
        background_color = cfg["btnclr"]        
        img[:] = background_color

    def drawThePage(self, pageName, img):


        buttons = self.pages[pageName]["buttons"]
        btnclr = cfg["btnclr"]
        R = cfg["btnparams"]["R"]
        txt = cfg["txtparams"]
      
        xadj,yadj,font,fontScale,tness = itemgetter("xadj","yadj","font","fontScale","thickness")(txt)
        txtclr = cfg["txtclr"]


        coords = self.getCoordintates() if "coords" not in cfg else cfg["coords"]

        for i in range(0, len(buttons)):
            if buttons[i] and buttons[i].strip():
                cv2.rectangle(img, coords[i][0], coords[i][1], btnclr, cv2.FILLED)
                cv2.putText(img, buttons[i], (coords[i][0][0] + xadj, coords[i][1][1] + yadj),
                            font, fontScale, txtclr, tness)
                cv2.circle(img, coords[i][2], R, btnclr, 2)
            if "pagetitle" in self.pages[pageName]:
                title,titleY,fontSize,titleclr,titletness = self.pages[pageName]["pagetitle"]
                titleX = self.getXOrgofText(title,font,fontSize,titletness)
                while titleX >= (cfg["screen_x"]+50):
                    fontSize = fontSize - 0.25
                    titleX = self.getXOrgofText(title,font,fontSize,titletness)
                cv2.putText(img, title, (titleX, titleY), font, fontSize, titleclr, titletness)

    def getXOrgofText(self, text, fontFace, fontScale, thickness):
        (W, H), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
        rem = cfg["screen_x"] - W
        return (rem-1)//2 if rem%2 == 1 else rem//2


    def getCoordintates(self):        
        
        btnpr = cfg["btnparams"]
        W, H, BtnSp, CirSp = btnpr["W"], btnpr["H"], btnpr["BtnSp"],  btnpr["CirSp"]
        scr_x = cfg["screen_x"]
        scr_y = cfg["screen_y"]
        coords = []

        
        pt1 = (0, scr_y - (4*H + 3*BtnSp))
        pt2 = (W, scr_y - (3*H + 3*BtnSp))
        c = (W + CirSp, pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (scr_x - W, scr_y - (4*H + 3*BtnSp))
        pt2 = (scr_x, scr_y - (3*H + 3*BtnSp))
        c = (pt1[0] - CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (0, scr_y - (3*H + 2*BtnSp))
        pt2 = (W, scr_y - (2*H + 2*BtnSp))
        c = (W + CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (scr_x - W, scr_y - (3*H + 2*BtnSp))
        pt2 = (scr_x, scr_y - (2*H + 2*BtnSp))
        c = (pt1[0] - CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (0, scr_y - (2*H + BtnSp))
        pt2 = (W, scr_y - (H + BtnSp))
        c = (W + CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (scr_x - W, scr_y - (2*H + BtnSp))
        pt2 = (scr_x, scr_y - (H + BtnSp))
        c = (pt1[0] - CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (0, (scr_y - H))
        pt2 = (W, scr_y)
        c = (W + CirSp, pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        pt1 = (scr_x - W, scr_y - H)
        pt2 = (scr_x, scr_y)
        c = (pt1[0] - CirSp , pt2[1] - (H//2))
        coords.append((pt1, pt2, c))

        cfg["coords"] = coords

        return coords
