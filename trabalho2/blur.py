# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import cv2
from time import time
import typing


def measure_runtime(func: typing.Callable) -> typing.Callable:
    def wrapper(img: cv2.typing.MatLike, width: int, height: int) -> typing.Any:
        t1 = time()
        result = func(img, width, height)
        print(f"Tempo de execução da função {func.__name__}: {time() - t1}s")
        return result

    return wrapper


@measure_runtime
def blur_naive_algorithm(
    img: cv2.typing.MatLike, width: int, height: int
) -> cv2.typing.MatLike:
    window_size = width * height

    img_out = img.copy()
    for row in range(height // 2, len(img) - height // 2):
        for col in range(width // 2, len(img[row]) - width // 2):
            for channel in range(len(img[row][col])):
                sum_ = 0
                for window_row in range(row - height // 2, row + height // 2 + 1):
                    for window_col in range(col - width // 2, col + width // 2 + 1):
                        sum_ += img[window_row][window_col][channel]

                mean = sum_ / window_size

                img_out[row][col][channel] = mean
    return img_out


@measure_runtime
def blur_separable_filter(
    img: cv2.typing.MatLike, width: int, height: int
) -> cv2.typing.MatLike:
    img_step_1 = img.copy()

    # horizontal
    for row in range(height // 2, len(img) - height // 2):
        for channel in range(len(img[row][0])):
            sum_ = 0
            # first sum
            for window_col in range(0, width):
                sum_ += img[row][window_col][channel]
            img_step_1[row][width // 2][channel] = sum_ / width

            for window_col in range(width, len(img[row])):
                sum_ += (
                    -img[row][window_col - width][channel]
                    + img[row][window_col][channel]
                )
                img_step_1[row][window_col - width // 2][channel] = sum_ / width

    # vertical
    img_out = img_step_1.copy()
    for col in range(width // 2, len(img) - width // 2):
        for channel in range(len(img[0][col])):
            sum_ = 0
            # first sum
            for window_row in range(0, height):
                sum_ += img_step_1[window_row][col][channel]
            img_out[height // 2][col][channel] = sum_ / height

            for window_row in range(height, len(img)):
                sum_ += (
                    -img_step_1[window_row - height][col][channel]
                    + img_step_1[window_row][col][channel]
                )
                img_out[window_row - height // 2][col][channel] = sum_ / height

    return img_out


def build_integral_image(img: cv2.typing.MatLike):
    integral_image = img.copy()

    # horizontal
    for row in range(len(img)):
        for col in range(1, len(img[row])):
            for channel in range(len(img[row][col])):
                integral_image[row][col][channel] += integral_image[row][col - 1][
                    channel
                ]

    # vertical
    for row in range(1, len(img)):
        for col in range(len(img[row])):
            for channel in range(len(img[row][col])):
                integral_image[row][col][channel] += integral_image[row - 1][col][
                    channel
                ]

    return integral_image


@measure_runtime
def blur_integral_image(
    img: cv2.typing.MatLike,
    width: int,
    height: int,
) -> cv2.typing.MatLike:
    integral_image = build_integral_image(img)
    img_out = img.copy()

    W, H = width // 2, height // 2

    # sem borda
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            for channel in range(img.shape[2]):
                t, b = max(0, row - H - 1), min(row + H, img.shape[0] - 1)
                l, r = max(0, col - W - 1), min(col + W, img.shape[1] - 1)
                value = (
                    integral_image[b][r][channel]
                    - integral_image[t][r][channel]
                    - integral_image[b][l][channel]
                    + integral_image[t][l][channel]
                ) / ((b - t) * (r - l))
                img_out[row][col][channel] = value

    return img_out


algorithms = {
    "ingenuo": blur_naive_algorithm,
    "separavel": blur_separable_filter,
    "integral": blur_integral_image,
}
