import cv2
import numpy as np

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')

# Convert to float32 for scaling
image_float = np.float32(image)

# Contrast factor
contrast = 1.5  # Increase contrast
bright = 0      # No change in brightness

# Apply contrast
enhanced = cv2.convertScaleAbs(image_float, alpha=contrast, beta=bright)

cv2.imshow("Original Image", image)
cv2.imshow("Contrast Enhanced", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
