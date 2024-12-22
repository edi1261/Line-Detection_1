import cv2 as cv
import numpy as np

lower_white = np.array([0, 0, 155])
upper_white = np.array([255, 155, 255])

low_green = np.array([35, 100, 100])
up_green = np.array([85, 255, 255])


def calculate_angle(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def get_field_mask(frame, low_green, up_green):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    green_mask = cv.inRange(hsv, low_green, up_green)
    green_mask_eroded = cv.erode(green_mask, kernel, iterations=1)
    green_mask_dilated = cv.dilate(green_mask_eroded, kernel, iterations=6)
    return green_mask_dilated

def find_largest_contour(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        return largest_contour
    return None

def extract_roi(frame, largest_contour):
    hull = cv.convexHull(largest_contour)
    mask = np.zeros_like(frame[:, :, 0])
    cv.drawContours(mask, [hull], -1, 255, thickness=-1)
    roi_frame = cv.bitwise_and(frame, frame, mask=mask)
    return roi_frame, hull

def process_frame(frame, low_green, up_green):
    field_mask = get_field_mask(frame, low_green, up_green)
    largest_contour = find_largest_contour(field_mask)
    if largest_contour is not None:
        roi_frame, hull = extract_roi(frame, largest_contour)
        return field_mask, roi_frame, hull
    return field_mask, None, None

def calculate_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1 + 1e-6)

def is_parallel_and_close(line1, line2, slope_threshold=0.1, distance_threshold=10):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    slope1 = calculate_slope(x1, y1, x2, y2)
    slope2 = calculate_slope(x3, y3, x4, y4)

    if abs(slope1 - slope2) > slope_threshold:
        return False

    dist = abs((y3 - slope1 * x3 - (y1 - slope1 * x1)) / np.sqrt(1 + slope1**2))
    return dist < distance_threshold

def merge_lines(lines):
    merged_lines = []

    while lines:
        line = lines.pop(0)
        x1, y1, x2, y2 = line[0]
        max_line = line
        max_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        to_remove = []
        for other_line in lines:
            x3, y3, x4, y4 = other_line[0]
            if is_parallel_and_close((x1, y1, x2, y2), (x3, y3, x4, y4)):
                length = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
                if length > max_length:
                    max_length = length
                    max_line = other_line
                to_remove.append(other_line)

        for line_to_remove in to_remove:
            lines.remove(line_to_remove)

        merged_lines.append(max_line)
    return merged_lines

def intersect(frame, lines, frame_counter, frame_skip=1):
    
    if frame_counter % frame_skip != 0:
        return    
    
    if lines is None:
        return
    i = 0
    for line in lines:
        i += 1
        L1, T1, L2, T2 = 0, 0, 0, 0

        x1, y1, x2, y2 = line[0]
        r = 8
        text = "line" + str(i)
       
        cv.circle(frame, (x1, y1), 3, (0, 255, 0), -1)
        cv.circle(frame, (x2, y2), 3, (0, 255, 0), -1)
        cv.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

        x11, y11 = max(x1 - r, 0), max(y1 - r, 0)
        x12, y12 = min(x1 + r, frame.shape[1]), min(y1 + r, frame.shape[0])
        surrounding_point1 = frame[y11:y12, x11:x12]

        x21, y21 = max(x2 - r, 0), max(y2 - r, 0)
        x22, y22 = min(x2 + r, frame.shape[1]), min(y2 + r, frame.shape[0])
        surrounding_point2 = frame[y21:y22, x21:x22]

        cv.rectangle(frame, (x11, y11), (x12, y12), (0, 255, 255), 1)
        cv.rectangle(frame, (x21, y21), (x22, y22), (0, 255, 255), 1)

        is_point1 = cv.inRange(surrounding_point1, (240, 0, 0), (255, 10, 10))
        is_line1 = cv.inRange(surrounding_point1, (0, 0, 240), (10, 10, 255))
        L1 = np.sum(is_point1)
        T1 = np.sum(is_line1)
        
        th= 1200

        if L1 > th:
            cv.putText(frame, "L1", (x1 - r, y1 + r + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif T1 >th:
            cv.putText(frame, "T1", (x1 - r, y1 + r + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        is_point2 = cv.inRange(surrounding_point2, (240, 0, 0), (255, 10, 10))
        is_line2 = cv.inRange(surrounding_point2, (0, 0, 240), (10, 10, 255))
        L2 = np.sum(is_point2)
        T2 = np.sum(is_line2)
        
        if L2 > th:
            cv.putText(frame, "L2", (x2 - r, y2 + r + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        elif T2 > th:
            cv.putText(frame, "T2", (x2 - r, y2 + r + 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        print(text, "point 1 =", L1, "line 1 =", T1, "point 2 =", L2, "line 2 =", T2)

        cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
        cv.circle(frame, (x2, y2), 3, (255, 0, 0), -1)

video_path = "sweeping.mp4"
cap = cv.VideoCapture(video_path)

if not cap.isOpened():
    print("Video tidak dapat dibuka. Pastikan path benar.")
else:
    frame_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        field_mask, roi_frame, hull = process_frame(frame, low_green, up_green)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        if roi_frame is not None:
            hsv = cv.cvtColor(roi_frame, cv.COLOR_BGR2HSV)
            blurred_bin = cv.GaussianBlur(hsv, (5, 5), 2)
            bin_image = cv.inRange(blurred_bin, lower_white, upper_white)
            edge = cv.Canny(bin_image, 50, 150)
            dilated_edges = cv.dilate(edge, kernel, iterations=1)

            lines = cv.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=100, minLineLength=175, maxLineGap=40)
            if lines is not None:
                merged_lines = merge_lines(lines.tolist())
                for line in merged_lines:
                    x1, y1, x2, y2 = line[0]
                    cv.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv.circle(frame, (x1, y1), 3, (255, 0, 0), -1)
                    cv.circle(frame, (x2, y2), 3, (255, 0, 0), -1)

#frame skip pramaeter can be determmined
            intersect(frame, lines, frame_counter, frame_skip=30)

            cv.imshow('Processed Video', frame)

        if cv.waitKey(100) & 0xFF == ord('q'):
            break
        
        frame_counter += 1

    cap.release()
    cv.destroyAllWindows()
