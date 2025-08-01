import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, convolve2d

img = cv2.imread(R'D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))
#edge kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

correlated = correlate2d(img, kernel, boundary='symm', mode='same')

convolved = convolve2d(img, kernel, boundary='symm', mode='same')

# Normalize for display
correlated = np.clip(correlated, 0, 255).astype(np.uint8)
convolved = np.clip(convolved, 0, 255).astype(np.uint8)

#results
plt.figure(figsize=(10, 4))
titles = ['Original', 'Correlation', 'Convolution']
images = [img, correlated, convolved]

for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
