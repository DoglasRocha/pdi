# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import cv2
from time import time
import typing


def measure_runtime(func: typing.Callable) -> typing.Callable:
    t1 = time()

    def wrapper(img: cv2.typing.MatLike, w1: int, w2: int) -> typing.Any:
        result = func(img, w1, w2)
        print(f"Tempo de execução da função {func.__name__}: {time() - t1}")
        return result

    return wrapper


@measure_runtime
def naive_algorithm(img: cv2.typing.MatLike, w1: int, w2: int) -> cv2.typing.MatLike:
    img_out = img.copy()
    for row in range(w2 // 2, len(img) - w2 // 2):
        for col in range(w1 // 2, len(img[row]) - w1 // 2):
            for channel in range(len(img[row][col])):
                sum_ = 0
                for window_row in range(row - w2 // 2, row + w2 // 2 + 1):
                    for window_col in range(col - w1 // 2, col + w1 // 2 + 1):
                        sum_ += img[window_row][window_col][channel]

                mean = sum_ / (w1 * w2)

                img_out[row][col][channel] = mean
    return img_out


@measure_runtime
def separable_filter(img: cv2.typing.MatLike, w1: int, w2: int) -> cv2.typing.MatLike:
    pass


@measure_runtime
def integral_image(img: cv2.typing.MatLike, w1: int, w2: int) -> cv2.typing.MatLike:
    pass


algorithms = {
    "ingenuo": naive_algorithm,
    "separavel": separable_filter,
    "integral": integral_image,
}
