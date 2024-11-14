import cv2
import os
from image_utils import to_grayscale, apply_gaussian_blur, detect_edges, find_contours


# function to filter through contours and keep typical trading card contours
# default aspect ratio is set
def filter_contours(contours, min_area=5000, min_aspect_ratio=1.4, max_aspect_ratio=1.8):


    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Get the bounding rectangle for aspect ratio calculation
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0  # Avoid division by zero
        
        # Filter based on area and aspect ratio
        if area > min_area and min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            filtered_contours.append(contour)
    
    return filtered_contours


# function to process the single image takes in processed image and where image will output too
def process_single_image(image_path, output_folder):

    image = cv2.imread(image_path)
    
    gray_image = to_grayscale(image)
    blurred_image = apply_gaussian_blur(gray_image)
    edge_image = detect_edges(blurred_image)
    
    contours = find_contours(edge_image)
    filtered_contours = filter_contours(contours)  # Only keep card-sized contours
    
    # Return the filtered contours for verification
    return filtered_contours, image, edge_image
