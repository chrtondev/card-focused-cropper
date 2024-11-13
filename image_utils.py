import cv2

# function to convert image to grayscale, takes in color image returns grayscale imgae
def to_grayscale(image):
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# function to apply a gaussian blur to the card to reduce noise, takes in grayscale image returns blur image, kernel size is default 
def apply_gaussian_blur(image, kernel_size=5):
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# function to detect edges in the image using the canary edge detector
def detect_edges(image, lower_threshold=50, upper_threshold=150):

    return cv2.Canny(image, lower_threshold, upper_threshold)

# funciton to find the contours in the image that is edge detected
def find_contours(edge_image):

    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
