import cv2
import numpy as np
from matplotlib import pyplot as plt

def split_merge(img, threshold):
    h, w = img.shape
    mean, std = cv2.meanStdDev(img)
    if std < threshold or h < 8 or w < 8:
        mean_val = np.ones((h, w), dtype=np.uint8) * int(mean)
        return mean_val
    
    h2, w2 = h // 2, w // 2
    top_left = split_merge(img[:h2, :w2], threshold)
    top_right = split_merge(img[:h2, w2:], threshold)
    bottom_left = split_merge(img[h2:, :w2], threshold)
    bottom_right = split_merge(img[h2:, w2:], threshold)

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))
    merged = np.vstack((top, bottom))
    
    return merged

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)
blur = cv2.GaussianBlur(img, (5, 5), 0)

segmented = split_merge(blur, threshold=10)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title("Original Image"), plt.imshow(img, cmap='gray')
plt.subplot(1,2,2), plt.title("Segmented (Split & Merge)"), plt.imshow(segmented, cmap='gray')
plt.show()
