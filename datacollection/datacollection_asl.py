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
folder="data/ASL/"

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
    
    if (key == ord("a") ):
        count+=1
        cv2.imwrite(f'{folder}A/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("b") ):
        count+=1
        cv2.imwrite(f'{folder}B/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("c") ):
        count+=1
        cv2.imwrite(f'{folder}C/{str(count)}.jpg',whiteimg)
        print(count)
    
    elif (key == ord("d") ):
        count+=1
        cv2.imwrite(f'{folder}D/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("e") ):
        count+=1
        cv2.imwrite(f'{folder}E/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("f") ):
        count+=1
        cv2.imwrite(f'{folder}F/{str(count+35)}.jpg',whiteimg)
        print(count)

    elif (key == ord("g") ):
        count+=1
        cv2.imwrite(f'{folder}G/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("h") ):
        count+=1
        cv2.imwrite(f'{folder}H/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("i") ):
        count+=1
        cv2.imwrite(f'{folder}I/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("j") ):
        count+=1
        cv2.imwrite(f'{folder}J/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("k") ):
        count+=1
        cv2.imwrite(f'{folder}K/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("l") ):
        count+=1
        cv2.imwrite(f'{folder}L/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("m") ):
        count+=1
        cv2.imwrite(f'{folder}M/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("n") ):
        count+=1
        cv2.imwrite(f'{folder}N/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("o") ):
        count+=1
        cv2.imwrite(f'{folder}O/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("p") ):
        count+=1
        cv2.imwrite(f'{folder}P/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("q") ):
        count+=1
        cv2.imwrite(f'{folder}Q/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("r") ):
        count+=1
        cv2.imwrite(f'{folder}R/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("s") ):
        count+=1
        cv2.imwrite(f'{folder}S/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("t") ):
        count+=1
        cv2.imwrite(f'{folder}T/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("u") ):
        count+=1
        cv2.imwrite(f'{folder}U/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("v") ):
        count+=1
        cv2.imwrite(f'{folder}V/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("w") ):
        count+=1
        cv2.imwrite(f'{folder}W/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("x") ):
        count+=1
        cv2.imwrite(f'{folder}X/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("y") ):
        count+=1
        cv2.imwrite(f'{folder}Y/{str(count+59)}.jpg',whiteimg)
        print(count)

    elif (key == ord("z") ):
        count+=1
        cv2.imwrite(f'{folder}Z/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("1") ):
        count+=1
        cv2.imwrite(f'{folder}Space/{str(count)}.jpg',whiteimg)
        print(count)

    elif (key == ord("2") ):
        count+=1
        cv2.imwrite(f'{folder}Period/{str(count)}.jpg',whiteimg)
        print(count)