import cv2 
import numpy as np

low_white = np.array([0, 0, 165])
up_white = np.array([300, 60, 255])

low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])

cap = cv2.VideoCapture('recording1.mp4')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blurred_bin = cv2.GaussianBlur(hsv, (5,5), 2)
    bin_image = cv2.inRange(blurred_bin, low_green, up_green)
    edge = cv2.Canny(bin_image, 50, 150)
    dilated_edges = cv2.dilate(edge, kernel, iterations=1)

      
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    # Display the result
    cv2.imshow('morph open', bin_image)
    cv2.imshow('ori', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
