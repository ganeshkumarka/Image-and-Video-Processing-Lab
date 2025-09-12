import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# DFT without Padding
dft1 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft1_shift = np.fft.fftshift(dft1)

# low-pass filter mask
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols, 2), np.uint8)
r = 30   # radius for low-pass filter
cv2.circle(mask, (ccol, crow), r, (1, 1, 1), -1)

# Apply mask
fshift1 = dft1_shift * mask
f_ishift1 = np.fft.ifftshift(fshift1)
img_back1 = cv2.idft(f_ishift1)
img_back1 = cv2.magnitude(img_back1[:, :, 0], img_back1[:, :, 1])

# DFT with Padding 
padded = np.zeros((2*rows, 2*cols), np.float32)
padded[:rows, :cols] = img  

dft2 = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
dft2_shift = np.fft.fftshift(dft2)

# Create low-pass mask for padded image
mask2 = np.zeros((2*rows, 2*cols, 2), np.uint8)
cv2.circle(mask2, (cols, rows), r, (1, 1, 1), -1)

# Apply mask
fshift2 = dft2_shift * mask2
f_ishift2 = np.fft.ifftshift(fshift2)
img_back2 = cv2.idft(f_ishift2)
img_back2 = cv2.magnitude(img_back2[:, :, 0], img_back2[:, :, 1])

# Crop back to original size
img_back2 = img_back2[:rows, :cols]

# results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.axis("off")

plt.subplot(1, 3, 2), plt.imshow(img_back1, cmap='gray')
plt.title("DFT Without Padding"), plt.axis("off")

plt.subplot(1, 3, 3), plt.imshow(img_back2, cmap='gray')
plt.title("DFT With Padding"), plt.axis("off")

plt.show()
