import cv2
import numpy as np 
import math as m
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

def ASLDetector():
    detector = HandDetector(maxHands=1)  # detector object
    classifier = Classifier("model/ASL/keras_model_200epoch_right_hand.h5", "model/ASL/labels_right_hand.txt")  # classifier object
    cap = cv2.VideoCapture(0)  # capture object
    offset = 20  # img offset
    imgSize = 500  # box size

    labels = ("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"," ",".")
    dot_detected_time = None  # variable to track the time when "." is detected
    countdown_time = 5  # countdown duration in seconds
    
    while True:
        success, img = cap.read()  # success is a boolean variable and img is an array of every pixel
        if success:
            imgOutput = img.copy()
            imgOutput = cv2.resize(imgOutput, (1000, 700))  # Resize to 1000x1000 pixels
            hands, img = detector.findHands(img)  # hands is the list of all hands
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                whiteimg = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # white background

                if x > offset and y > offset:
                    imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                    aspectratio = h / w
                    if aspectratio >= 1:  # if height > width
                        k = imgSize / h
                        wCal = m.ceil(k * w)
                        imgResize = cv2.resize(imgcrop, (wCal, imgSize), interpolation=cv2.INTER_AREA)
                        if imgResize.shape[1] > 0 and imgResize.shape[1] <= 500:
                            wGap = m.ceil((imgSize - wCal) / 2)
                            whiteimg[:, wGap:wCal + wGap] = imgResize
                            prediction, index = classifier.getPrediction(whiteimg)
                            detected_label = labels[index]
                            print(detected_label)

                    else:
                        k = imgSize / w
                        hCal = m.ceil(k * h)
                        imgResize = cv2.resize(imgcrop, (imgSize, hCal), interpolation=cv2.INTER_AREA)
                        if imgResize.shape[0] > 0 and imgResize.shape[0] <= 500:
                            hGap = m.ceil((imgSize - hCal) / 2)
                            whiteimg[hGap:hCal + hGap, :] = imgResize
                            prediction, index = classifier.getPrediction(whiteimg)
                            detected_label = labels[index]
                            print(detected_label)

                    # Display the language indicator with smaller text
                    cv2.putText(imgOutput, "Current Language Set Is: ASL", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
                    cv2.putText(imgOutput, detected_label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
                    cv2.imshow("whiteimg", whiteimg)

                    # Check for "." detection
                    if detected_label == ".":
                        if dot_detected_time is None:
                            dot_detected_time = time.time()  # start timing when "." is first detected
                        else:
                            elapsed_time = time.time() - dot_detected_time
                            remaining_time = countdown_time - elapsed_time
                            if remaining_time > 0:
                                # Display countdown message with smaller text
                                cv2.putText(imgOutput, f"Program terminates in {int(remaining_time)} seconds", (20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                            elif elapsed_time >= countdown_time:  # if 5 seconds have passed
                                print("Detected '.' for 5 seconds. Exiting ASLDetector.")
                                break
                    else:
                        dot_detected_time = None  # reset timer if "." is not detected

        cv2.imshow("myimg", imgOutput)
        key = cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

while True:
    lang = input("Enter the lang needed: ")
    if int(lang) == 1:
        ASLDetector()
