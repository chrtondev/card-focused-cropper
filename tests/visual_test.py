import cv2
from image_utils import to_grayscale, apply_gaussian_blur, detect_edges, find_contours


# this is a test to see the process at each step and see everything work, to help fine tune settings etc


# Load a sample image
image = cv2.imread('data/input/sample_scan.png')  # Update the path if your test image has a different name or format

# Step 1: Convert to grayscale
gray_image = to_grayscale(image)
cv2.imshow("Grayscale Image", gray_image)
cv2.waitKey(0)

# Step 2: Apply Gaussian blur
blurred_image = apply_gaussian_blur(gray_image)
cv2.imshow("Blurred Image", blurred_image)
cv2.waitKey(0)

# Step 3: Detect edges
edge_image = detect_edges(blurred_image)
cv2.imshow("Edge Detected Image", edge_image)
cv2.waitKey(0)

# Step 4: Find and display contours
contours = find_contours(edge_image)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # Draw contours in green
cv2.imshow("Contours", contour_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
