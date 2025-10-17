import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)

# Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(img, (5, 5), 0)

# Apply Otsu's thresholding 
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

binary = cv2.bitwise_not(binary)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title("Original Image"), plt.imshow(img, cmap='gray')
plt.subplot(1,2,2), plt.title("Segmented Blobs (Thresholding)"), plt.imshow(binary, cmap='gray')
plt.show()
