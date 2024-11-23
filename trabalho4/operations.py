import cv2, numpy as np
from img_utils import reshape_image
from segmentation import binarize


def supress_image_noise(
    img: cv2.typing.MatLike,
    kernel: "np.ndarray" = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
) -> cv2.typing.MatLike:
    kernel = kernel.astype(np.uint8)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    img = reshape_image(img)
    return img


def normalize_locally(
    img: cv2.typing.MatLike, kernel_size: "tuple[int, int]" = (200, 200)
) -> cv2.typing.MatLike:
    blurry = cv2.blur(img, kernel_size)
    blurry = reshape_image(blurry)

    # img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = reshape_image(img)
    return img - blurry


def normalize_locally_and_binarize_image(
    img: cv2.typing.MatLike, threshold: float = 0.18
) -> cv2.typing.MatLike:
    normalized = normalize_locally(img)
    binarized = binarize(normalized, threshold)
    binarized = reshape_image(binarized)

    return binarized


def sharp_image(
    img: cv2.typing.MatLike, kernel_size: "tuple[int, int]" = (13, 13)
) -> cv2.typing.MatLike:
    blurry = cv2.blur(img, kernel_size)
    blurry = reshape_image(blurry)

    details = img - blurry

    sharped_image = img + details

    return sharped_image
