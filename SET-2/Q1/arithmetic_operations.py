import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg')
img2 = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-2\lenna.png')

# Resize to same dimensions if needed
img1 = cv2.resize(img1, (256, 256))
img2 = cv2.resize(img2, (256, 256))

# Arithmetic operations
add = cv2.add(img1, img2)
subtract = cv2.subtract(img1, img2)
multiply = cv2.multiply(img1, img2)
divide = cv2.divide(img1, img2)

# results
titles = ['Image 1', 'Image 2', 'Addition', 'Subtraction', 'Multiplication', 'Division']
images = [img1, img2, add, subtract, multiply, divide]

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
