import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim
    HAVE_SSIM = True
except Exception:
    HAVE_SSIM = False

def add_salt_and_pepper_noise(image, amount=0.02):
    noisy = image.copy()
    h, w = image.shape
    num_pixels = h * w
    num_salt = int(np.ceil(amount * num_pixels * 0.5))
    num_pepper = int(np.ceil(amount * num_pixels * 0.5))

    coords = (np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt))
    noisy[coords] = 255

    coords = (np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper))
    noisy[coords] = 0

    return noisy

def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 10 * np.log10((PIXEL_MAX ** 2) / mse)

INPUT_PATH = 'D:\Image-and-Video-Processing-Lab\SET-2\elephant.jpg'
OUTSIZE = (256, 256)
SP_AMOUNT = 0.05             
KERNEL = 3                      

img = cv2.imread(INPUT_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not load {INPUT_PATH}")
img = cv2.resize(img, OUTSIZE)


noisy = add_salt_and_pepper_noise(img, amount=SP_AMOUNT)

median_clean = cv2.medianBlur(img, KERNEL)
median_noisy = cv2.medianBlur(noisy, KERNEL)

cv2.imwrite('q5_clean.png', img)
cv2.imwrite('q5_clean_median.png', median_clean)
cv2.imwrite('q5_noisy.png', noisy)
cv2.imwrite('q5_noisy_median.png', median_noisy)

psnr_noisy = psnr(img, noisy)
psnr_clean_median = psnr(img, median_clean)
psnr_noisy_median = psnr(img, median_noisy)

print(f"PSNR(original, noisy)..........: {psnr_noisy:.2f} dB")
print(f"PSNR(original, clean_median)....: {psnr_clean_median:.2f} dB")
print(f"PSNR(original, noisy_median)....: {psnr_noisy_median:.2f} dB")

if HAVE_SSIM:
    ssim_noisy = ssim(img, noisy)
    ssim_clean_median = ssim(img, median_clean)
    ssim_noisy_median = ssim(img, median_noisy)
    print(f"SSIM(original, noisy)..........: {ssim_noisy:.4f}")
    print(f"SSIM(original, clean_median)....: {ssim_clean_median:.4f}")
    print(f"SSIM(original, noisy_median)....: {ssim_noisy_median:.4f}")
else:
    print("scikit-image not found — SSIM not computed (optional).")

# results
titles = [
    'Original (clean)',
    f'Clean + Median ({KERNEL}x{KERNEL})',
    f'Noisy (S&P {int(SP_AMOUNT*100)}%)',
    f'Noisy + Median ({KERNEL}x{KERNEL})'
]
images = [img, median_clean, noisy, median_noisy]

plt.figure(figsize=(12, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
