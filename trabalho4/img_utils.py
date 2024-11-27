import cv2, numpy as np

def open_image(img_path: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = reshape_image(img)
    img = img.astype(np.float32) / 255.0

    return img


def reshape_image(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img = img.reshape((img.shape[0], img.shape[1], 1))
    return img