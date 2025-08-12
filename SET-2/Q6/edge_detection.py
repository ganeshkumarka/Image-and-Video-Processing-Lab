import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Sobe
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(np.absolute(sobel_combined))

#Prewitt
prewitt_kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], dtype=np.float32)
prewitt_kernel_y = np.array([[-1, -1, -1],
                             [ 0,  0,  0],
                             [ 1,  1,  1]], dtype=np.float32)
prewitt_x = cv2.filter2D(img, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(img, -1, prewitt_kernel_y)
prewitt_combined = cv2.add(prewitt_x, prewitt_y)

#Roberts Cross
roberts_cross_v = np.array([[1, 0],
                            [0, -1]], dtype=np.float32)
roberts_cross_h = np.array([[0, 1],
                            [-1, 0]], dtype=np.float32)
roberts_v = cv2.filter2D(img, -1, roberts_cross_v)
roberts_h = cv2.filter2D(img, -1, roberts_cross_h)
roberts_combined = cv2.add(roberts_v, roberts_h)

#Laplacian
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

#Laplacian of Gaussian
gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
log = cv2.Laplacian(gaussian_blur, cv2.CV_64F)
log = np.uint8(np.absolute(log))

#Canny
canny_edges = cv2.Canny(img, 100, 200)

#results
titles = ['Original', 'Sobel', 'Prewitt', 'Roberts', 'Laplacian', 'LoG', 'Canny']
images = [img, sobel_combined, prewitt_combined, roberts_combined, laplacian, log, canny_edges]

plt.figure(figsize=(12, 8))
for i in range(len(images)):
    plt.subplot(3, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
