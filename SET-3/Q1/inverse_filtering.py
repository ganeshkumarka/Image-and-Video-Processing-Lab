import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)

# Simulate degradation (motion blur kernel)
kernel = np.zeros((15, 15))
kernel[7] = np.ones(15) / 15
degraded = cv2.filter2D(img, -1, kernel)

# Fourier transform
G = np.fft.fft2(degraded)
H = np.fft.fft2(kernel, s=img.shape)

# Direct Inverse Filtering
eps = 1e-5 
F_hat = G / (H + eps)
restored = np.abs(np.fft.ifft2(F_hat))

# results
plt.subplot(1,3,1), plt.title("Original"), plt.imshow(img, cmap='gray')
plt.subplot(1,3,2), plt.title("Degraded"), plt.imshow(degraded, cmap='gray')
plt.subplot(1,3,3), plt.title("Restored"), plt.imshow(restored, cmap='gray')
plt.show()
