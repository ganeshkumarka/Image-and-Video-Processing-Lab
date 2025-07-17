import cv2

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')

# Complement (Negative)
complement = 255 - image

# Show result
cv2.imshow("Original Image", image)
cv2.imshow("Complement Image", complement)
cv2.waitKey(0)
cv2.destroyAllWindows()
