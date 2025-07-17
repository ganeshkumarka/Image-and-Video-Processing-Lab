import cv2
import matplotlib.pyplot as plt

image_path = r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Create CLAHE object (clipLimit controls contrast, tileGridSize divides the image into tiles)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Apply CLAHE
local_eq = clahe.apply(image)

# Plot original vs local histogram equalized
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Local Histogram Equalized")
plt.imshow(local_eq, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
