# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import cv2, numpy as np


def get_windows(passes: int, sigma: float):
    """
    Fast Almost-Gaussian Filtering The Australian Pattern Recognition Society Conference: DICTA 2010. December 2010. Sydney.
    https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
    """
    w_ideal = ((12 * (sigma**2)) / passes + 1) ** (1 / 2)
    wl = int(((w_ideal - 1) // 2) * 2 + 1)
    wu = wl + 2

    n = passes
    m = ((12 * sigma**2) - (n * wl**2) - (4 * n * wl) - (3 * n)) / (-4 * wl - 4)
    m = round(m)

    windows = []
    for _ in range(m):
        windows.append((wl, wl))

    for _ in range(n - m):
        windows.append((wu, wu))

    return windows


IMG_PATH = "img/Wind Waker GC.bmp"
THRESHOLD_LUMINANCE = 0.5

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

blurred_g_masks = []
blurred_b_masks = []
for i in range(3, 7):
    sigma = i**2
    blurred_g_masks.append(cv2.GaussianBlur(mask.copy(), (0, 0), sigma))

    tmp = mask.copy()
    for window in get_windows(4, sigma):
        tmp = cv2.blur(tmp, window)
    blurred_b_masks.append(tmp)

sum_g_mask = np.zeros(img.shape)
sum_b_mask = np.zeros(img.shape)
for g_mask, b_mask in zip(blurred_g_masks, blurred_b_masks):
    sum_g_mask += g_mask
    sum_b_mask += b_mask

cv2.imshow("mascara gaussiana", sum_g_mask)
cv2.imshow("mascara box blur", sum_b_mask)
cv2.waitKey()
cv2.destroyAllWindows()

# final_g_mask = np.clip(sum_g_mask, 0, 1)
# final_b_mask = np.clip(sum_b_mask, 0, 1)
final_g_mask = sum_g_mask
final_b_mask = sum_b_mask

bloom_g_img = 0.92 * img + 0.08 * final_g_mask
bloom_b_img = 0.92 * img + 0.08 * final_b_mask

cv2.imshow("original", img)
cv2.imshow("bloom com filtro gaussiano", bloom_g_img)
cv2.imshow("bloom com filtro da media", bloom_b_img)
cv2.waitKey()
cv2.destroyAllWindows()
