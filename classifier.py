import cv2
import numpy as np 
import math as m
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap=cv2.VideoCapture(0)#capture object
detector=HandDetector(maxHands=1) #detector object
classifier=Classifier("model/ASL/keras_model_200epoch_right_hand.h5","modelASL//labels_right_hand.txt") #classifier object

offset=20 #img offset
imgSize=500 #box size
folder="data/"

labels=("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," ",".")

while(True):
    success,img =cap.read()#success is boolean variable and img is array of every pixel 
    if(success):
        imgOutput=img.copy()
        hands,img= detector.findHands(img) #hands is the list of all hands 
        if hands:
            hand=hands[0]
            x,y,w,h = hand['bbox']
            whiteimg=np.ones((imgSize,imgSize,3),np.uint8)*255 #white bg
            if(x>offset and y>offset):
                imgcrop=img[y-offset:y+h+offset,x-offset:x+w+offset] #img is basically a matrix with height and width paramters with stating and ending ones offset is an offset  

                handShape=imgcrop.shape #.shape is a matrix with 3 colunns height width and channel

                aspectratio=h/w
                if(aspectratio>=1): #if height >width
                    k=imgSize/h
                    wCal=m.ceil((k*w))
                    imgResize=cv2.resize(imgcrop,(wCal,imgSize),interpolation=cv2.INTER_AREA)
                    imgResizeShape=imgResize.shape
                    if(imgResize.shape[1]>0 and imgResize.shape[1]<=500):
                        wGap=m.ceil((imgSize-wCal)/2)
                        whiteimg[:,wGap:wCal+wGap]=imgResize #fixing hand onto white bg
                        prediction,index=classifier.getPrediction(whiteimg)
                        print(labels[index])
                        
                
                else:
                    k=imgSize/w
                    hCal=m.ceil((k*h))
                    imgResize=cv2.resize(imgcrop,(imgSize,hCal),interpolation=cv2.INTER_AREA)
                    imgResizeShape=imgResize.shape
                    if(imgResize.shape[0]>0 and imgResize.shape[0]<=500):
                        hGap=m.ceil((imgSize-hCal)/2)
                        whiteimg[hGap:hCal+hGap,:]=imgResize #fixing hand onto white bg
                        prediction,index=classifier.getPrediction(whiteimg)
                        print(labels[index])

                cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,0),2)

                cv2.imshow("croppedhand",imgcrop) 
                cv2.imshow("whiteimg",whiteimg)


    cv2.imshow("myimg",imgOutput)
    key=cv2.waitKey(1)

