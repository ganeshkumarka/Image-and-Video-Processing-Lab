import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)

# Motion blur kernel
kernel = np.zeros((15, 15))
kernel[7] = np.ones(15) / 15
degraded = cv2.filter2D(img, -1, kernel)

# Fourier transforms
G = np.fft.fft2(degraded)
H = np.fft.fft2(kernel, s=img.shape)

# Constant ratio Wiener filter
K = 0.01
F_hat1 = (np.conj(H) / (np.abs(H)**2 + K)) * G
restored1 = np.abs(np.fft.ifft2(F_hat1))

# Using autocorrelation (approximated as noise variance)
Sn = 0.001
Sf = np.abs(np.fft.fft2(img))**2
F_hat2 = (np.conj(H) / (np.abs(H)**2 + (Sn/Sf))) * G
restored2 = np.abs(np.fft.ifft2(F_hat2))

plt.subplot(1,3,1), plt.title("Degraded"), plt.imshow(degraded, cmap='gray')
plt.subplot(1,3,2), plt.title("Wiener (const K)"), plt.imshow(restored1, cmap='gray')
plt.subplot(1,3,3), plt.title("Wiener (auto corr)"), plt.imshow(restored2, cmap='gray')
plt.show()
