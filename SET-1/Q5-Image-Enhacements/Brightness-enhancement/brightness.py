import cv2
import numpy as np

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')
# Increase brightness by 50
brightness_value = 50
bright_image = cv2.add(image, np.ones(image.shape, dtype='uint8') * brightness_value)

cv2.imshow("Original Image", image)
cv2.imshow("Brightened Image", bright_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
