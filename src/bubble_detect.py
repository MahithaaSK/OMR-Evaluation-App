import cv2
import numpy as np

def find_bubble_contours(thresh_img, min_area=300, max_area=5000):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4 * np.pi * (area / (peri*peri))
        if circularity < 0.4:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bubbles.append({'contour': c, 'bbox': (x, y, w, h), 'center': (x + w//2, y + h//2)})
    return bubbles
