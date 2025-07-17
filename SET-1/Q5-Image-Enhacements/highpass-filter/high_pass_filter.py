import cv2
import numpy as np

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg', cv2.IMREAD_GRAYSCALE)

# Define high-pass filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Apply filter
high_pass = cv2.filter2D(image, -1, kernel)

cv2.imshow("Original", image)
cv2.imshow("High-Pass Filtered", high_pass)
cv2.waitKey(0)
cv2.destroyAllWindows()
