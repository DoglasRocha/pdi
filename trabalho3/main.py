import cv2, numpy as np

IMG_PATH = "img/Wind Waker GC.bmp"
THRESHOLD_LUMINANCE = 0.525

# IMG_PATH = "img/GT2.BMP"
# THRESHOLD_LUMINANCE = 0.5

img = cv2.imread(IMG_PATH).astype(np.float32) / 255.0
img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)

mask = img.copy()
for row in range(img.shape[0]):
    for col in range(img.shape[1]):
        if img_hls[row, col, 1] < THRESHOLD_LUMINANCE:
            mask[row, col, :] = 0

# cv2.imshow("mask", mask)

blurred_masks = []
for sigma in [2, 4, 8, 16]:
    blurred_masks.append(cv2.GaussianBlur(mask.copy(), (0, 0), sigma))

final_mask = np.zeros(img.shape)
for mask in blurred_masks:
    final_mask += mask

final_mask = np.clip(final_mask, 0, 1)
bloom_img = 0.92 * img + 0.08 * final_mask
cv2.imshow("original", img)
cv2.imshow("bloom", bloom_img)
cv2.waitKey()
