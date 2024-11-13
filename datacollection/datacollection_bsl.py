import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector

# Base path for storing images
base_folder = "Simulation-Modelling_7THSEMPROJ/data/BSL/"

# Ask the user to enter a letter to specify the folder
label = input("Enter the letter (e.g., 'A' for folder A) to save images: ").upper()
folder = os.path.join(base_folder, label)

# Create the specified folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

cap = cv2.VideoCapture(0)  # Capture object
detector = HandDetector(maxHands=2)  # Detector object for two hands

offset = 20  # Image offset
imgSize = 500  # Size of each hand image
canvasSize = 1000  # Size of the background canvas
count = 0  # Initialize image count
capture_started = False  # Flag to start capturing
start_time = 0  # Initialize start time for delay
countdown = 10  # Countdown in seconds

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)  # Detect two hands

    # Countdown display
    if capture_started and countdown > 0:
        cv2.putText(img, f"Capturing in {countdown} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if len(hands) == 2:  # Check if two hands are detected
        # Get bounding boxes for both hands
        hand1 = hands[0]
        hand2 = hands[1]
        x1, y1, w1, h1 = hand1['bbox']
        x2, y2, w2, h2 = hand2['bbox']

        # Create a 1000x1000 white background image
        whiteimg = np.ones((canvasSize, canvasSize, 3), np.uint8) * 255

        # Crop and resize each hand region to fit into a 500x500 box
        def crop_and_resize_hand(imgcrop, target_size=imgSize):
            h, w, _ = imgcrop.shape
            aspectratio = h / w
            if aspectratio >= 1:  # Height >= Width
                k = target_size / h
                imgResize = cv2.resize(imgcrop, (int(w * k), target_size))
                hGap = (target_size - imgResize.shape[1]) // 2
                imgWhite = np.ones((target_size, target_size, 3), np.uint8) * 255
                imgWhite[:, hGap:hGap + imgResize.shape[1]] = imgResize
            else:  # Width > Height
                k = target_size / w
                imgResize = cv2.resize(imgcrop, (target_size, int(h * k)))
                vGap = (target_size - imgResize.shape[0]) // 2
                imgWhite = np.ones((target_size, target_size, 3), np.uint8) * 255
                imgWhite[vGap:vGap + imgResize.shape[0], :] = imgResize
            return imgWhite

        # Place each hand image on the larger 1000x1000 canvas
        hand1_img = crop_and_resize_hand(img[y1 - offset:y1 + h1 + offset, x1 - offset:x1 + w1 + offset])
        hand2_img = crop_and_resize_hand(img[y2 - offset:y2 + h2 + offset, x2 - offset:x2 + w2 + offset])
        whiteimg[0:imgSize, 0:imgSize] = hand1_img
        whiteimg[0:imgSize, imgSize:canvasSize] = hand2_img

        cv2.imshow("Both Hands on White Canvas", whiteimg)  # Display combined hand image

    cv2.imshow("Live Feed", img)
    key = cv2.waitKey(1)

    # Start the 10-second countdown when "s" is pressed and two hands are detected
    if key == ord("s"):
        capture_started = True
        start_time = time.time()  # Record start time
        count = 0  # Reset count for new sequence
        countdown = 10  # Reset countdown

    # If capturing has started, update the countdown
    if capture_started:
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:  # Start capturing after 10 seconds
            if count < 100:
                count += 1
                cv2.imwrite(f'{folder}/BothHands_{str(count)}.jpg', whiteimg)
                print(f"Saved image {count} in folder: {folder}")
            else:
                capture_started = False  # Stop capturing after 100 images
        else:
            # Update countdown display
            countdown = max(0, 10 - int(elapsed_time))

    # Press "q" to exit the loop
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
