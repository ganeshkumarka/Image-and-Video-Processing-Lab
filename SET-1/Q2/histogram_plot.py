import cv2
import matplotlib.pyplot as plt
import os

# Load image using absolute path
image_path = r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram using OpenCV
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Plot using Matplotlib
plt.figure(figsize=(8, 5))
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(hist, color='black')
plt.xlim([0, 256])
plt.grid(True)
plt.show()
