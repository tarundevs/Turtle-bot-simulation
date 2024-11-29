import cv2
import numpy as np

# Create a blank image
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Define colors
blue = (255, 0, 0)
red = (0, 0, 255)

# Draw blue cone shape (triangle)
pts_blue = np.array([[250, 100], [200, 400], [300, 400]], np.int32)
pts_blue = pts_blue.reshape((-1, 1, 2))
cv2.fillPoly(image, [pts_blue], blue)

# Draw red cone shape (triangle)
pts_red = np.array([[150, 200], [100, 400], [200, 400]], np.int32)
pts_red = pts_red.reshape((-1, 1, 2))
cv2.fillPoly(image, [pts_red], red)

# Convert image to HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges in HSV
lower_blue = np.array([100, 150, 0])
upper_blue = np.array([140, 255, 255])

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

# Create masks for blue and red colors
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Find contours for blue color
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_blue:
    cv2.drawContours(image, [cnt], -1, (255, 255, 255), 2)

# Find contours for red color
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours_red:
    cv2.drawContours(image, [cnt], -1, (255, 255, 255), 2)

# Display the image
cv2.imshow('Image', image)
cv2.imshow('Blue Mask', mask_blue)
cv2.imshow('Red Mask', mask_red)
cv2.waitKey(0)
cv2.destroyAllWindows()
