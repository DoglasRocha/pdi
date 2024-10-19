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
def naive_algorithm(
    img: cv2.typing.MatLike, width: int, height: int
) -> cv2.typing.MatLike:
    img_out = img.copy()
    for row in range(height // 2, len(img) - height // 2):
        for col in range(width // 2, len(img[row]) - width // 2):
            for channel in range(len(img[row][col])):
                sum_ = 0
                for window_row in range(row - height // 2, row + height // 2 + 1):
                    for window_col in range(col - width // 2, col + width // 2 + 1):
                        sum_ += img[window_row][window_col][channel]

                mean = sum_ / (width * height)

                img_out[row][col][channel] = mean
    return img_out


@measure_runtime
def separable_filter(
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


@measure_runtime
def integral_image(
    img: cv2.typing.MatLike, width: int, height: int
) -> cv2.typing.MatLike:
    pass


algorithms = {
    "ingenuo": naive_algorithm,
    "separavel": separable_filter,
    # "integral": integral_image,
}
