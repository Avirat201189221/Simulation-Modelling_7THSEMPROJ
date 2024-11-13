import cv2
import numpy as np 
import math as m
from cvzone.HandTrackingModule import HandDetector
# from datetime import datetime
# import random as r

cap=cv2.VideoCapture(0)#capture object

detector=HandDetector(maxHands=1) #detector object

offset=20 #img offset
imgSize=500 #box size
folder="data/General_Classifier/"

count=0

while(True):
    success,img =cap.read()
    hands,img= detector.findHands(img) #hands is the list of all hands 
    if (hands):
        hand=hands[0]
        x,y,w,h = hand['bbox']

        whiteimg=np.ones((imgSize,imgSize,3),np.uint8)*255 #white bg
        if(x>offset and y>offset):
            imgcrop=img[y-offset:y+h+offset,x-offset:x+w+20] #img is basically a matrix with height and width paramters with stating and ending ones offset is an offset  

            # handShape=imgcrop.shape #.shape is a matrix with 3 colunns height width and channel

            aspectratio=h/w
            if(aspectratio>=1): #if height >=width
                k=imgSize/h
                wCal=m.ceil((k*w))
                imgResize=cv2.resize(imgcrop,(wCal+1,imgSize))
                if(imgResize.shape[1]>0 and imgResize.shape[1]<=500):
                    imgResizeShape=imgResize.shape
                    wGap=m.ceil((imgSize-wCal+1)/2)
                    whiteimg[:,wGap:wCal+1+wGap]=imgResize #fixing hand onto white bg
            
            else:
                k=imgSize/w
                hCal=m.ceil((k*h))
                imgResize=cv2.resize(imgcrop,(imgSize,hCal+1))
                if(imgResize.shape[0]>0 and imgResize.shape[0]<=500):
                    imgResizeShape=imgResize.shape
                    hGap=m.ceil((imgSize-hCal+1)/2)
                    whiteimg[hGap:hCal+1+hGap,:]=imgResize #fixing hand onto white bg

            cv2.imshow("croppedhand",imgcrop) 
            cv2.imshow("whiteimg",whiteimg)


    cv2.imshow("myimg",img)
    key=cv2.waitKey(1)
    

    if (key == ord("1") ):
        count+=1
        cv2.imwrite(f'{folder}One/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("2") ):
        count+=1
        cv2.imwrite(f'{folder}Two/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("3") ):
        count+=1
        cv2.imwrite(f'{folder}Three/{str(count)}.jpg',whiteimg)
        print(count)
    
    elif (key == ord("4") ):
        count+=1
        cv2.imwrite(f'{folder}Four/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("5") ):
        count+=1
        cv2.imwrite(f'{folder}Period/{str(count)}.jpg',whiteimg)
        print(count)