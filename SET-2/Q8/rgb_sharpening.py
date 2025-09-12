import cv2
import numpy as np

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg')
img = cv2.resize(img, (256, 256))

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

sharpened = cv2.subtract(img, laplacian)

cv2.imshow("Original RGB", img)
cv2.imshow("Laplacian Edges", laplacian)
cv2.imshow("Sharpened RGB", sharpened)
cv2.waitKey(0)
cv2.destroyAllWindows()
