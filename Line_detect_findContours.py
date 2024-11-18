import cv2
import numpy as np
import time

# Define color ranges
low_white = np.array([0, 0, 165])
up_white = np.array([300, 60, 255])

low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])

low_yellow = np.array([20, 100, 100])
up_yellow = np.array([40, 255, 255])

vid = cv2.VideoCapture('recording1.mp4')

# Parameters
min_line_length = 250  # Minimum line length to display
frame_rate = 2
frame_count = 0
fps_counter = time.time()

while True:
    ret, frame = vid.read()
    if not ret:
        break

    start_time = time.time()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    if frame_count % frame_rate == 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks
        green_mask = cv2.inRange(hsv, low_green, up_green)
        morph_close = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edge = cv2.Canny(morph_close, 50, 150)
        contours, _ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
        # Draw detected lines with length filter
        for contour in contours:
            length = cv2.arcLength(contour, False)
            if length > min_line_length:
                cv2.polylines(frame, [contour], isClosed=False, color=(255, 0, 0), thickness=2)

        # Show frames
        cv2.imshow('Line Detection', frame)
        cv2.imshow('Edge Detection', edge)

    frame_count += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
