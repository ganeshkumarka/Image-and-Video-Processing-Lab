import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_mean_filter(image, ksize=3):
    return cv2.blur(image, (ksize, ksize))

def apply_box_filter(image, ksize=3):
    return cv2.boxFilter(image, -1, (ksize, ksize), normalize=True)

def apply_gaussian_filter(image, ksize=5, sigma=1):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)

img = cv2.imread(r"D:\Image-and-Video-Processing-Lab\SET-2\Q2\gaussian_noisy.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

mean_filtered = apply_mean_filter(img, 3)
box_filtered = apply_box_filter(img, 3)
gaussian_filtered = apply_gaussian_filter(img, 5, 1)

# results
titles = ['Original', 'Mean Filter', 'Box Filter', 'Gaussian Filter']
images = [img, mean_filtered, box_filtered, gaussian_filtered]

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
