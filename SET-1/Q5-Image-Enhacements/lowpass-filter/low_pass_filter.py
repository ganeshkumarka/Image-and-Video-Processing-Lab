import cv2

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')
# Apply low-pass filter (Gaussian Blur)
blur = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imshow("Original", image)
cv2.imshow("Low-Pass Filtered", blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
