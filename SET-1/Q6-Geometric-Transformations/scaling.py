import cv2

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')

scaled = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

cv2.imshow("Original", image)
cv2.imshow("Scaled", scaled)
cv2.waitKey(0)
cv2.destroyAllWindows()
