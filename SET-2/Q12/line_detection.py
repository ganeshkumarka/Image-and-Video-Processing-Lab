import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# filter masks
horizontal_mask = np.array([[-1, -1, -1],
                            [ 2,  2,  2],
                            [-1, -1, -1]])

vertical_mask = np.array([[-1,  2, -1],
                          [-1,  2, -1],
                          [-1,  2, -1]])

diagonal_mask = np.array([[ 2, -1, -1],
                          [-1,  2, -1],
                          [-1, -1,  2]])

# Apply filters
horizontal_lines = cv2.filter2D(img, -1, horizontal_mask)
vertical_lines = cv2.filter2D(img, -1, vertical_mask)
diagonal_lines = cv2.filter2D(img, -1, diagonal_mask)

# results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title("Original"), plt.axis("off")

plt.subplot(2, 2, 2), plt.imshow(horizontal_lines, cmap='gray')
plt.title("Horizontal Lines"), plt.axis("off")

plt.subplot(2, 2, 3), plt.imshow(vertical_lines, cmap='gray')
plt.title("Vertical Lines"), plt.axis("off")

plt.subplot(2, 2, 4), plt.imshow(diagonal_lines, cmap='gray')
plt.title("Diagonal Lines"), plt.axis("off")

plt.show()
