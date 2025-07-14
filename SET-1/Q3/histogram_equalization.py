import cv2
import matplotlib.pyplot as plt

# Load image
image_path = r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg' 
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized = cv2.equalizeHist(image)

# Plot original and equalized histograms
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Original Histogram")
plt.hist(image.ravel(), bins=256, range=[0, 256], color='blue')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.title("Equalized Histogram")
plt.hist(equalized.ravel(), bins=256, range=[0, 256], color='green')
plt.grid(True)

plt.tight_layout()
plt.show()
