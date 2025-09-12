import cv2
import numpy as np
from matplotlib import pyplot as plt

def rgb_to_hsi(img):
    img = img.astype(np.float32) / 255
    R, G, B = cv2.split(img)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 2*np.pi - theta) / (2*np.pi)
    S = 1 - (3/(R+G+B+1e-6)) * np.min([R,G,B], axis=0)
    I = (R+G+B) / 3

    return (H, S, I)

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg")
H, S, I = rgb_to_hsi(img)

plt.subplot(1,3,1), plt.title("Hue"), plt.imshow(H, cmap='gray')
plt.subplot(1,3,2), plt.title("Saturation"), plt.imshow(S, cmap='gray')
plt.subplot(1,3,3), plt.title("Intensity"), plt.imshow(I, cmap='gray')
plt.show()
