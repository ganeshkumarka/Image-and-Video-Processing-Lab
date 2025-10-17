import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import deque

def region_growing(img, seed, threshold=10):

    h, w = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    seed_val = img[seed]
    
    q = deque([seed])
    mask[seed] = 255

    #connected neighbors
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

    while q:
        x, y = q.popleft()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                if abs(int(img[nx, ny]) - int(seed_val)) < threshold:
                    mask[nx, ny] = 255
                    q.append((nx, ny))
    return mask

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)
blur = cv2.GaussianBlur(img, (5,5), 0)

seed_point = (img.shape[0]//2, img.shape[1]//2)

segmented = region_growing(blur, seed_point, threshold=15)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title("Original Image"), plt.imshow(img, cmap='gray')
plt.subplot(1,2,2), plt.title("Segmented Region"), plt.imshow(segmented, cmap='gray')
plt.show()
