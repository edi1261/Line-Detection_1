import cv2 
import numpy as np


low_white = np.array([0, 0, 165])
up_white = np.array([300, 60, 255])

low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])

low_yellow = np.array([20, 100, 100])
up_yellow = np.array([40, 255, 255])

    
vid = cv2.VideoCapture('recording1.mp4')  


def detect_lines(binary_image, frame):
    points = []
    height, width = binary_image.shape
    for y in range(height):
        for x in range(1, width - 1):
            if binary_image[y, x-1] == 0 and binary_image[y, x] == 255 and binary_image[y, x+1] == 0:
                points.append((x, y))

    if points:
        points = sorted(points)
        connected_points = [points[0]]
        for point in points[1:]:
            last_point = connected_points[-1]
            if abs(point[0] - last_point[0]) < 10 and abs(point[1] - last_point[1]) < 10:
                connected_points.append(point)
            else:
                if len(connected_points) > 5:
                    cv2.polylines(frame, [np.array(connected_points)], isClosed=False, color=(255, 0, 0), thickness=2)
                connected_points = [point]
        if len(connected_points) > 5:
            cv2.polylines(frame, [np.array(connected_points)], isClosed=False, color=(255, 0, 0), thickness=2)
    return frame, connected_points

# Read frame at an interval to reduce processing
frame_rate = 2
frame_count = 0

while True:
    ret, frame = vid.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    
    if frame_count % frame_rate == 0:
       kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
       green_mask = cv2.inRange(hsv, low_green, up_green)
       result = cv2.bitwise_and(frame, frame, mask=green_mask)
       morph_close = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
       edge = cv2.Canny(morph_close, 50, 150)
       dilated_edges = cv2.dilate(edge, kernel, iterations=1)
       frame_with_lines, lines = detect_lines(edge, frame)
       print(lines)
       cv2.imshow('Line Detection', frame_with_lines)
       cv2.imshow('edge1', edge)
    
    frame_count += 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()