import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def mousePoint(event, x,y,flags,params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x,y)

img = cv.imread('clg.jpg')
cv.imshow("IMG",img)
cv.setMouseCallback("IMG",mousePoint)
cv.waitKey(0)