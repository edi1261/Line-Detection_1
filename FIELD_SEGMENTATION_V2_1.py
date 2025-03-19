import cv2 as cv
import numpy as np

vid = cv.VideoCapture('recording1.mp4')

while True:
    isTrue, imgframe = vid.read()
    
    if not isTrue:
        break 

    # Konversi ke HSV
    hsv = cv.cvtColor(imgframe, cv.COLOR_BGR2HSV)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

    # Deteksi warna dominan
    hist = cv.calcHist([hsv], [0], None, [180], [0, 180])  # Hitung histogram hue
    dominant_hue = np.argmax(hist)  # Temukan warna dominan
    low_green = np.array([dominant_hue - 10, 100, 100])  # Sesuaikan rentang hijau
    up_green = np.array([dominant_hue + 10, 255, 255])
    
    # Deteksi warna hijau
    green_mask = cv.inRange(hsv, low_green, up_green)

    # Gabungkan dengan mask putih (opsional jika garis putih masih mengganggu)
    low_white = np.array([0, 0, 200])
    up_white = np.array([180, 30, 255])
    white_mask = cv.inRange(hsv, low_white, up_white)
    combined_mask = cv.bitwise_or(green_mask, white_mask)
    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=4)

    # Deteksi kontur
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv.contourArea)
        hull = cv.convexHull(largest_contour)

        # Buat mask untuk ROI
        mask = np.zeros_like(combined_mask)
        cv.drawContours(mask, [hull], -1, 255, thickness=-1)

        # Gambarkan boundary pada frame asli
        boundary_frame = imgframe.copy()
        cv.drawContours(boundary_frame, [hull], -1, (0, 255, 0), thickness=2)

        # Tampilkan hasil
        roi_frame = cv.bitwise_and(imgframe, imgframe, mask=mask)
        cv.imshow('Field Only', boundary_frame)
        cv.imshow('ROI Frame', roi_frame)
        cv.imshow('Filled Mask', combined_mask)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()