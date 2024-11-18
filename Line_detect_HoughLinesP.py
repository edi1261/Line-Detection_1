import cv2 
import numpy as np
from collections import deque


# Parameters
step_size = 8 # Step size for scanning
white_ratio_threshold = 1.0  # Ratio to determine if points are on the same line
error_threshold = 20  # Threshold for the error function

def generate_points(image, step_size):
    points = []
    rows, cols = image.shape
    for y in range(0, rows, step_size):
        for x in range(0, cols, step_size):
            if x + 2 * step_size < cols:
                if (image[y, x] == 0 and image[y, x + step_size] == 255 and image[y, x + 2 * step_size] == 0):
                    points.append((x + step_size, y))
    return points

def find_best_candidate(start_point, points):
    min_dist = float('inf')
    best_point = None
    for point in points:
        dist = np.linalg.norm(np.array(point) - np.array(start_point))
        if dist < min_dist:
            min_dist = dist
            best_point = point
    return best_point

def compute_error(line, candidate):
    start_point = np.array(line[0])
    end_point = np.array(line[-1])
    candidate_point = np.array(candidate)
    line_vec = end_point - start_point
    point_vec = candidate_point - start_point
    error = np.abs(np.cross(line_vec, point_vec) / np.linalg.norm(line_vec))
    return error

def line_segment_detection(image, points):
    line_segments = []
    while points:
        line = deque([points.pop(0)])  
        while points:
            best_candidate = find_best_candidate(line[-1], points)
            points.remove(best_candidate)
            error = compute_error(line, best_candidate)
            if error < error_threshold:
                line.append(best_candidate)
            else:
                line_segments.append(list(line))  
                line.clear()
                break
    return line_segments

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
    morph_close = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel)
    edge = cv2.Canny(morph_close, 50, 150)
    dilated_edges = cv2.dilate(edge, kernel, iterations=1)

    
    points = generate_points(dilated_edges, step_size)
    line_segments = line_segment_detection(dilated_edges, points)
    
    # Draw detected line segments
    for segment in line_segments:
        for i in range(len(segment) - 1):
            cv2.line(frame, segment[i], segment[i + 1], 255, 5)
    
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=150, maxLineGap=10)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    
    # Display the result
    cv2.imshow("morph close", morph_close)
    cv2.imshow('morph open', bin_image)
    cv2.imshow('ori', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
