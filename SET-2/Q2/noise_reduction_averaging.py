import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('gaussian_noisy.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

averaging_counts = [2, 8, 16, 32, 128]
averaged_images = []

for count in averaging_counts:
    stack = np.stack([img for _ in range(count)], axis=0)
    avg_img = np.mean(stack, axis=0).astype(np.uint8)
    averaged_images.append(avg_img)

#results
plt.figure(figsize=(12, 6))
for i, avg_img in enumerate(averaged_images):
    plt.subplot(2, 3, i+1)
    plt.imshow(avg_img, cmap='gray')
    plt.title(f'Averaged {averaging_counts[i]}x')
    plt.axis('off')

plt.tight_layout()
plt.show()
