import cv2
import numpy as np

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg', cv2.IMREAD_GRAYSCALE)

# Define range to slice (highlight)
lower = 100
upper = 150

# Brightness slicing
sliced = np.where((image >= lower) & (image <= upper), 255, image)

cv2.imshow("Original", image)
cv2.imshow("Brightness Sliced", np.uint8(sliced))
cv2.waitKey(0)
cv2.destroyAllWindows()
