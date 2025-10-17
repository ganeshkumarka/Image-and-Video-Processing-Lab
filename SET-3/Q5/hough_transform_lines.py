import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg", 0)

edges = cv2.Canny(img, 100, 200)

lines = cv2.HoughLinesP(edges,
                        rho=1,                # distance resolution in pixels
                        theta=np.pi/180,      # angle resolution in radians
                        threshold=100,        # min number of votes
                        minLineLength=50,     # minimum length of line to detect
                        maxLineGap=10)        # max gap between line segments

output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1), plt.title("Edge Image"), plt.imshow(edges, cmap='gray')
plt.subplot(1,2,2), plt.title("Detected Lines"), plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.show()
