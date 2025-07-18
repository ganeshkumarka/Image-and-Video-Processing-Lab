import cv2
import numpy as np

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')
rows, cols = image.shape[:2]

# Skew matrix: horizontal skew (shear in x-direction)
M = np.float32([[1, 0.5, 0],
                [0,   1, 0]])
# Apply skewing
skewed = cv2.warpAffine(image, M, (int(cols * 1.5), rows))

cv2.imshow("Original", image)
cv2.imshow("Skewed", skewed)
cv2.waitKey(0)
cv2.destroyAllWindows()

