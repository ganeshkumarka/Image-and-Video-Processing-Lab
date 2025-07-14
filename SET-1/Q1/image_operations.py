import cv2
import os

image_path =  r'd:\Image-and-Video-Processing-Lab\SET-1\elephant.jpg'
image = cv2.imread(image_path)

# a. Read an image
cv2.imshow("Original Image", image)

# b. Get image information
print("Image Shape:", image.shape)
print("Image Size (Bytes):", image.size)
print("Image Datatype:", image.dtype)

# c. Find compression ratio
original_size = os.path.getsize(image_path)
compressed_path = 'compressed_image.jpg'
cv2.imwrite(compressed_path, image, [cv2.IMWRITE_JPEG_QUALITY, 50])
compressed_size = os.path.getsize(compressed_path)
compression_ratio = original_size / compressed_size
print(f"Compression Ratio: {compression_ratio:.2f}")

# d. Display negative of the image
negative = 255 - image
cv2.imshow("Negative Image", negative)

cv2.waitKey(0)
cv2.destroyAllWindows()
