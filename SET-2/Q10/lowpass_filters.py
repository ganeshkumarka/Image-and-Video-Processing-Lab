import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Compute DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Get image center
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

# Cutoff radius
D0 = 30

# Ideal Low Pass Filter (ILPF)
mask_ideal = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask_ideal, (ccol, crow), D0, (1, 1, 1), -1)

# Butterworth Low Pass Filter (BLPF)
n = 2  # order
mask_butter = np.zeros((rows, cols), np.float32)
for u in range(rows):
    for v in range(cols):
        D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
        mask_butter[u, v] = 1 / (1 + (D / D0) ** (2 * n))
mask_butter = cv2.merge([mask_butter, mask_butter])

# Gaussian Low Pass Filter (GLPF)
mask_gaussian = np.zeros((rows, cols), np.float32)
for u in range(rows):
    for v in range(cols):
        D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
        mask_gaussian[u, v] = np.exp(-(D ** 2) / (2 * (D0 ** 2)))
mask_gaussian = cv2.merge([mask_gaussian, mask_gaussian])

# Apply filters
def apply_filter(dft_shift, mask):
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    return cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

img_ideal = apply_filter(dft_shift, mask_ideal)
img_butter = apply_filter(dft_shift, mask_butter)
img_gaussian = apply_filter(dft_shift, mask_gaussian)

# results
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Original"), plt.axis("off")
plt.subplot(2, 2, 2), plt.imshow(img_ideal, cmap='gray'), plt.title("Ideal LPF"), plt.axis("off")
plt.subplot(2, 2, 3), plt.imshow(img_butter, cmap='gray'), plt.title("Butterworth LPF"), plt.axis("off")
plt.subplot(2, 2, 4), plt.imshow(img_gaussian, cmap='gray'), plt.title("Gaussian LPF"), plt.axis("off")
plt.show()
