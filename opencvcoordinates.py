import cv2

# Load the image in grayscale
image_path = r"\\wsl.localhost\Ubuntu-22.04\home\tarunwarrier\ros_ws\ERC-hackathon-2024\hackathon_automation\src\roads.png"  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Function to display the coordinates of the clicked point
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Print the coordinates on the console
        print(f"Coordinates: X: {x}, Y: {y}")
        # Display the coordinates on the image
        cv2.putText(image, f"({x},{y})", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('Image', image)

# Display the image
cv2.imshow('Image', image)

# Bind the click event to the window
cv2.setMouseCallback('Image', click_event)

# Wait for a key press and close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()


