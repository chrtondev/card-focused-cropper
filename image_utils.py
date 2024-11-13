import cv2

# function to convert image to grayscale, takes in color image returns grayscale imgae
def to_grayscale(image):
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# function to apply a gaussian blur to the card to reduce noise, takes in grayscale image returns blur image, kernel size is default 
def apply_gaussian_blur(image, kernel_size=5):
    
    return cv2.GaussianBlur(image, kernel_size, 0)

