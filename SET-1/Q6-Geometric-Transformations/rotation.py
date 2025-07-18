import cv2

image = cv2.imread(r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg')
rows, cols = image.shape[:2]

center = (cols // 2, rows // 2)
angle = 45 
scale = 1.0

M = cv2.getRotationMatrix2D(center, angle, scale)

rotated = cv2.warpAffine(image, M, (cols, rows))

cv2.imshow("Original", image)
cv2.imshow("Rotated", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
