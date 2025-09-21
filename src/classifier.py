import cv2
import numpy as np
import joblib

# Constants for bubble classification
CORNER_SIZE = 3
CENTER_REGION_RATIO = 0.6
INTENSITY_THRESHOLD = 0.8
DARK_PIXEL_RATIO = 0.7
MIN_DARK_PIXELS = 0.3

def load_model(model_path="models/bubble_model.pkl"):
    """Load the bubble classification model."""
    return joblib.load(model_path)

def get_background_intensity(crop):
    """Calculate background intensity from bubble corners.
    
    Args:
        crop (np.array): Cropped bubble image
        
    Returns:
        float: Average background intensity
    """
    corners = [
        crop[:CORNER_SIZE, :CORNER_SIZE],      # top-left
        crop[:CORNER_SIZE, -CORNER_SIZE:],     # top-right
        crop[-CORNER_SIZE:, :CORNER_SIZE],     # bottom-left
        crop[-CORNER_SIZE:, -CORNER_SIZE:]     # bottom-right
    ]
    return np.mean([np.mean(corner) for corner in corners])

def get_center_region(crop, width, height):
    """Extract the center region of the bubble.
    
    Args:
        crop (np.array): Cropped bubble image
        width (int): Bubble width
        height (int): Bubble height
        
    Returns:
        np.array: Center region of the bubble
    """
    margin = (1 - CENTER_REGION_RATIO) / 2
    center_y = int(height * margin)
    center_h = int(height * CENTER_REGION_RATIO)
    center_x = int(width * margin)
    center_w = int(width * CENTER_REGION_RATIO)
    
    return crop[center_y:center_y+center_h, center_x:center_x+center_w]

def classify_bubble(model, img_gray, bbox, img_size=(42,42)):
    """Classify a bubble as filled or unmarked.
    
    Args:
        model: Unused (kept for compatibility)
        img_gray (np.array): Grayscale image
        bbox (tuple): Bubble bounding box (x, y, width, height)
        img_size (tuple): Unused (kept for compatibility)
        
    Returns:
        str: 'filled', 'unmarked', or 'empty'
    """
    x, y, w, h = bbox
    crop = img_gray[y:y+h, x:x+w]
    
    if crop.size == 0:
        return 'empty'
    
    # Get background intensity from corners
    background = get_background_intensity(crop)
    
    # Get and analyze center region
    center_region = get_center_region(crop, w, h)
    mean_intensity = np.mean(center_region)
    intensity_ratio = mean_intensity / background
    
    # Calculate percentage of dark pixels
    dark_threshold = background * DARK_PIXEL_RATIO
    dark_pixels = np.sum(center_region < dark_threshold) / center_region.size
    
    # Classify based on darkness and dark pixel coverage
    if intensity_ratio < INTENSITY_THRESHOLD and dark_pixels > MIN_DARK_PIXELS:
        return 'filled'
    
    return 'unmarked'
