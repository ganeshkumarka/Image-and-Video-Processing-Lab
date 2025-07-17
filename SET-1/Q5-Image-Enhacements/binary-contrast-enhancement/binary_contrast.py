import cv2
image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg', cv2.IMREAD_GRAYSCALE)

# Apply threshold
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Grayscale", image)
cv2.imshow("Binary Contrast Image", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
