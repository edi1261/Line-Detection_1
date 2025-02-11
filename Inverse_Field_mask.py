import cv2 as cv
import numpy as np


vid = cv.VideoCapture('sweeping.mp4')


low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])


while True:
    isTrue, imgframe = vid.read()
    
    if not isTrue:
        break 
    
    hsv = cv.cvtColor(imgframe, cv.COLOR_BGR2HSV)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    green_mask = cv.inRange(hsv, low_green, up_green)
    green_mask1 = cv.erode(green_mask, kernel, iterations=1)
    green_mask2 = cv.dilate(green_mask1, kernel, iterations=6)
    inverse_green_mask = cv.bitwise_not(green_mask2)

    
    contours, _ = cv.findContours(inverse_green_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
       
    largest_contour = max(contours, key=cv.contourArea)
    
    # Buat mask untuk ROI
    mask = np.zeros_like(inverse_green_mask)
    cv.drawContours(mask, [largest_contour], -1, 255, thickness=-1)  # Isi area dalam hull

    # Gambarkan boundary pada frame asli
    boundary_frame = imgframe.copy()
    cv.drawContours(boundary_frame, [largest_contour], -1, (0, 0, 255), thickness=2)
    
    roi_frame = cv.bitwise_and(imgframe, imgframe, mask=mask)

    cv.imshow('Field Only', boundary_frame)
    cv.imshow('mask1', roi_frame)
    cv.imshow('mask', inverse_green_mask)

    
    if cv.waitKey(10) & 0xFF == ord('q'):  
        break

vid.release()
cv.destroyAllWindows()

    
    
    