import cv2
import numpy as np
from matplotlib import pyplot as plt

def rgb_to_hsi(img):
    img = img.astype(np.float32) / 255.0
    R, G, B = cv2.split(img)

    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + 1e-6
    theta = np.arccos(num / den)

    H = np.where(B <= G, theta, 2*np.pi - theta)
    H = H / (2 * np.pi)  
    S = 1 - (3 / (R + G + B + 1e-6)) * np.min([R, G, B], axis=0)
    I = (R + G + B) / 3

    return H, S, I

def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi
    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    idx = (H >= 0) & (H < 2*np.pi/3)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / (np.cos(np.pi/3 - H[idx])))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])

    idx = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    H2 = H[idx] - 2*np.pi/3
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H2)) / (np.cos(np.pi/3 - H2)))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])

    idx = (H >= 4*np.pi/3) & (H < 2*np.pi)
    H3 = H[idx] - 4*np.pi/3
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H3)) / (np.cos(np.pi/3 - H3)))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])

    rgb = cv2.merge((R, G, B))
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb

img = cv2.imread("D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg")
H, S, I = rgb_to_hsi(img)

# Histogram equalization on intensity
I_eq = cv2.equalizeHist((I * 255).astype(np.uint8))
I_eq = I_eq.astype(np.float32) / 255.0

# Convert equalized HSI back to RGB
rgb_eq = hsi_to_rgb(H, S, I_eq)

plt.figure(figsize=(10,5))
plt.subplot(1,3,1), plt.title("Original RGB"), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.subplot(1,3,2), plt.title("Equalized Intensity"), plt.imshow(I_eq, cmap='gray')
plt.subplot(1,3,3), plt.title("Equalized RGB Image"), plt.imshow(cv2.cvtColor(rgb_eq, cv2.COLOR_BGR2RGB))
plt.show()
