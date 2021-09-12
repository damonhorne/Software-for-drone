import cv2
import numpy as np

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX

while(1):

    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower boundary RED color range values; Hue (0 - 10)
    lower_red_1 = np.array([0, 120, 70])
    upper_red_1 = np.array([10, 255, 255])
 
    # upper boundary RED color range values; Hue (160 - 180)
    lower_red_2 = np.array([170,120,70])
    upper_red_2 = np.array([180,255,255])
    
    # combining red masks
    red_lower_mask = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_upper_mask = cv2.inRange(hsv, lower_red_2, upper_red_2)
    mask_red = red_lower_mask + red_upper_mask

    # boundary YELLOW color range values; Hue (20 - 30)
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # combining red and yellow masks
    mask = mask_red + mask_yellow
   
    kernel = np.ones((30, 30), np.uint8)
    kernel2 = np.ones((150, 150), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

    # Contours detection
    if int(cv2.__version__[0]) > 3:
        # Opencv 4.x.x
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # Opencv 3.x.x
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, False), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if area > 1000:
            cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)

            if 7 < len(approx) < 20:
                cv2.putText(frame, "Anaphylaxis", (x, y), font, 1, (0, 0, 0))
            elif len(approx) == 4:
                cv2.putText(frame, "Haemorrhage", (x, y), font, 1, (0, 0, 0))

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
   

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()