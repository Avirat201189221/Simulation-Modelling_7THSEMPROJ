import cv2
import numpy as np 
import math as m
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

def ASLDetector(cap):
    detector = HandDetector(maxHands=1)  # detector object
    classifier = Classifier("model/ASL/keras_model_200epoch_right_hand.h5", "model/ASL/labels_right_hand.txt")  # classifier object
     # capture object
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

    cv2.destroyWindow("myimg")
    cv2.destroyWindow("whiteimg")
def MSLDetector(cap):
    detector = HandDetector(maxHands=1)  # detector object
    classifier=Classifier("model/MSL/keras_model.h5","model/MSL/labels.txt")
     # capture object
    offset = 20  # img offset
    imgSize = 500  # box size

    labels=("A","B","C","D","E","F","G","H","I","J","K","L","LL","M","N","O","P","Q","R","RR","S","T","U","V","W","X","Y","Z",".")
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
                    cv2.putText(imgOutput, "Current Language Set Is: MSL", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
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
                                print("Detected '.' for 5 seconds. Exiting MSLDetector.")
                                break
                    else:
                        dot_detected_time = None  # reset timer if "." is not detected

        cv2.imshow("myimg", imgOutput)
        key = cv2.waitKey(1)

    cv2.destroyWindow("myimg")
    cv2.destroyWindow("whiteimg")
def LSFDetector(cap):
    detector = HandDetector(maxHands=1)  # detector object
    classifier=Classifier("model/LSF/keras_model.h5","model/LSF/labels.txt")
     # capture object
    offset = 20  # img offset
    imgSize = 500  # box size

    labels=("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",".")
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
                    cv2.putText(imgOutput, "Current Language Set Is: LSF", (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)
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
                                print("Detected '.' for 5 seconds. Exiting LSFDetector.")
                                break
                    else:
                        dot_detected_time = None  # reset timer if "." is not detected

        cv2.imshow("myimg", imgOutput)
        key = cv2.waitKey(1)

    cv2.destroyWindow("myimg")
    cv2.destroyWindow("whiteimg")

cap = cv2.VideoCapture(0) 
detector = HandDetector(maxHands=1)
classifier = Classifier("model/General_Classifier/keras_model.h5", "model/General_Classifier/labels.txt")

offset = 20
imgSize = 500
labels = ("1", "2", "3", "4", ".")
countdown_time = 5  # Countdown time for function activation

# Trackers for detection and countdown
dot_detected_time = None
number_detected_time = None
detected_number = None

while True:
    success, img = cap.read()
    if success:
        imgOutput = img.copy()
        imgOutput = cv2.resize(imgOutput, (1920, 1080))
        hands, img = detector.findHands(img)

        # Display instructions
        cv2.putText(imgOutput, "Hold 1 for ASL, 2 for MSL, 3 for LSF, 4 for BSL", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            whiteimg = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            if x > offset and y > offset:
                imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                aspectratio = h / w
                if aspectratio >= 1:
                    k = imgSize / h
                    wCal = m.ceil(k * w)
                    imgResize = cv2.resize(imgcrop, (wCal, imgSize), interpolation=cv2.INTER_AREA)
                    wGap = m.ceil((imgSize - wCal) / 2)
                    whiteimg[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = m.ceil(k * h)
                    imgResize = cv2.resize(imgcrop, (imgSize, hCal), interpolation=cv2.INTER_AREA)
                    hGap = m.ceil((imgSize - hCal) / 2)
                    whiteimg[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(whiteimg)
                detected_label = labels[index]
                print(detected_label)

                # Display detected label
                cv2.putText(imgOutput, detected_label, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 0), 2)
                
                # Check for "." to terminate the main program if held for 5 seconds
                if detected_label == ".":
                    if dot_detected_time is None:
                        dot_detected_time = time.time()
                    else:
                        elapsed_time = time.time() - dot_detected_time
                        remaining_time = countdown_time - elapsed_time
                        if remaining_time > 0:
                            cv2.putText(imgOutput, f"Program terminates in {int(remaining_time)} seconds", (20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                        elif elapsed_time >= countdown_time:
                            print("Detected '.' for 5 seconds. Exiting program.")
                            break
                else:
                    dot_detected_time = None  # Reset dot timer if "." is not detected

                # Check for numbers (1-4) to call specific detector functions if held for 5 seconds
                if detected_label in ("1", "2", "3", "4"):
                    if detected_number == detected_label:
                        elapsed_time = time.time() - number_detected_time
                        remaining_time = countdown_time - elapsed_time
                        if remaining_time > 0:
                            cv2.putText(imgOutput, f"Switching to language {detected_label} in {int(remaining_time)} seconds", (20, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        elif elapsed_time >= countdown_time:
                            if detected_label == "1":
                                print("Activating ASL Detector")
                                ASLDetector(cap)
                            elif detected_label == "2":
                                print("Activating MSL Detector")
                                MSLDetector(cap)
                            elif detected_label == "3":
                                print("Activating LSF Detector")
                                LSFDetector(cap)
                            # elif detected_label == "4":
                            #     print("Activating BSL Detector")
                            #     BSLDetector(cap)
                            # Do not break here; allow the main loop to resume after each detector finishes
                    else:
                        detected_number = detected_label
                        number_detected_time = time.time()
                else:
                    detected_number = None  # Reset number detection if another sign is detected

        cv2.imshow("Output", imgOutput)
        key = cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()