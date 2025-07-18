import cv2
import numpy as np


image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')
rows, cols = image.shape[:2]


M = np.float32([[1, 0, 100], [0, 1, 50]])

translated = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("Original", image)
cv2.imshow("Translated", translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
