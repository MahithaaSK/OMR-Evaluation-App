import pytesseract
import cv2

def extract_set_number(img_bgr, roi=None):
    """
    Extract handwritten 'Set No' from OMR sheet using Tesseract OCR
    roi: (x, y, w, h) tuple to crop Set No area. If None, use full image.
    """
    if roi:
        x, y, w, h = roi
        img = img_bgr[y:y+h, x:x+w]
    else:
        img = img_bgr.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789"
    text = pytesseract.image_to_string(gray, config=config)
    text = text.strip()
    return text
