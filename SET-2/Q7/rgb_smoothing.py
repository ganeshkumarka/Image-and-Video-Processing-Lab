import cv2
import numpy as np

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg')
img = cv2.resize(img, (256, 256))   
kernel = np.ones((3, 3), np.float32) / 9

smoothed = cv2.filter2D(img, -1, kernel)

# Show results
cv2.imshow("Original RGB", img)
cv2.imshow("Smoothed RGB", smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()
