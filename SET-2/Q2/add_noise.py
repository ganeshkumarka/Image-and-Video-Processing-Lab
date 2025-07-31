import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def add_gaussian_noise(image, mean=0, std=25):
    noise = np.random.normal(mean, std, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)

    # Add salt (white) noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy[tuple(coords)] = 255

    # Add pepper (black) noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy[tuple(coords)] = 0

    return noisy

# Load a clean grayscale image
img = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256))

# Add both Gaussian and Salt-and-Pepper noise
gaussian_noisy = add_gaussian_noise(img)
sp_noisy = add_salt_and_pepper_noise(img)

# Save noisy images for later use
cv2.imwrite('gaussian_noisy.jpg', gaussian_noisy)
cv2.imwrite('sp_noisy.jpg', sp_noisy)

# Show results
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gaussian_noisy, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sp_noisy, cmap='gray')
plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.tight_layout()
plt.show()
