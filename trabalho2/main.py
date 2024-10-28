# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import cv2, numpy as np
import os
from time import time
import typing


def measure_runtime(func: typing.Callable) -> typing.Callable:
    def wrapper(*args, **kwargs) -> typing.Any:
        t1 = time()
        result = func(*args, **kwargs)
        print(f"Tempo de execução da função {func.__name__}: {time() - t1}s")
        return result

    return wrapper


@measure_runtime
def blur_naive_algorithm(
    img: cv2.typing.MatLike, width: int, height: int, *args, **kwargs
) -> cv2.typing.MatLike:
    window_size = width * height

    img_out = img.copy()
    for row in range(height // 2, img.shape[0] - height // 2):
        for col in range(width // 2, img.shape[1] - width // 2):
            for channel in range(img.shape[2]):
                sum_ = 0
                for window_row in range(row - height // 2, row + height // 2 + 1):
                    for window_col in range(col - width // 2, col + width // 2 + 1):
                        sum_ += img[window_row][window_col][channel]

                mean = sum_ / window_size

                img_out[row][col][channel] = mean
    return img_out


@measure_runtime
def blur_separable_filter(
    img: cv2.typing.MatLike, width: int, height: int, *args, **kwargs
) -> cv2.typing.MatLike:
    img_step_1 = img.copy()

    # horizontal
    for row in range(0, img.shape[0]):
        for channel in range(img.shape[2]):
            sum_ = 0
            # first sum
            for window_col in range(0, width):
                sum_ += img[row][window_col][channel]
            img_step_1[row][width // 2][channel] = sum_ / width

            for window_col in range(width, img.shape[1]):
                sum_ += (
                    -img[row][window_col - width][channel]
                    + img[row][window_col][channel]
                )
                img_step_1[row][window_col - width // 2][channel] = sum_ / width

    # vertical
    img_out = img_step_1.copy()
    for col in range(img.shape[1]):
        for channel in range(img.shape[2]):
            sum_ = 0
            # first sum
            for window_row in range(0, height):
                sum_ += img_step_1[window_row][col][channel]
            img_out[height // 2][col][channel] = sum_ / height

            for window_row in range(height, img.shape[0]):
                sum_ += (
                    -img_step_1[window_row - height][col][channel]
                    + img_step_1[window_row][col][channel]
                )
                img_out[window_row - height // 2][col][channel] = sum_ / height

    return img_out


@measure_runtime
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
    integral_image: cv2.typing.MatLike | None = None,
    *args,
    **kwargs,
) -> cv2.typing.MatLike:
    if type(integral_image) == type(None):
        integral_image = build_integral_image(img)
    img_out = img.copy()

    W, H = width // 2, height // 2
    for row in range(0, img.shape[0]):
        for col in range(0, img.shape[1]):
            for channel in range(img.shape[2]):
                t, b = max(0, row - H - 1), min(row + H, img.shape[0] - 1)
                l, r = max(0, col - W - 1), min(col + W, img.shape[1] - 1)
                value = (
                    integral_image[b][r][channel]
                    - (integral_image[t][r][channel] if b - t > 0 else 0)
                    - (integral_image[b][l][channel] if r - l > 0 else 0)
                    + (integral_image[t][l][channel] if r - l > 0 and b - t > 0 else 0)
                ) / ((b - t if b - t > 0 else 1) * (r - l if r - l > 0 else 1))
                img_out[row][col][channel] = value

    return img_out


algorithms = {
    "ingenuo": blur_naive_algorithm,
    "separavel": blur_separable_filter,
    "com imagem integral": blur_integral_image,
}


def open_image(img_path: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path)
    if img is None:
        print("Erro ao abrir a imagem")
        exit(1)

    # float conversion
    img = img.astype(np.float32) / 255
    return img


def test(
    img_path: str,
    output_directory: str,
    output_name_template: str,
    window_sizes: list[tuple[int, int]],
    blur_algorithms: list,
    show_images: bool = False,
) -> None:
    print(f"==================== TESTE DA IMAGEM {img_path} ====================")
    os.makedirs(output_directory, exist_ok=True)

    img = open_image(img_path)
    integral_img = build_integral_image(img)
    for window_index, w_size in enumerate(window_sizes):
        print(f"------ TAMANHO DE JANELA: {w_size} ------")
        for name, func in blur_algorithms:
            result = func(img, *w_size, integral_img)
            cv2.imwrite(
                os.path.join(
                    output_directory,
                    f"{output_name_template}{window_index + 2} - Borrada {w_size} - algoritmo {name}.png",
                ),
                result * 255,
            )

            open_cv_right_image = cv2.blur(img, w_size)
            comparison = open_cv_right_image - result
            norm_comparison = cv2.normalize(comparison, None, 0, 1, cv2.NORM_MINMAX)
            if show_images:
                im_show = np.concatenate([result, comparison, norm_comparison], axis=1)
                cv2.imshow(
                    f"Resultado algoritmo {name} {w_size} VS Diferenca OpenCV e Resultado VS Diferenca normalizada",
                    im_show,
                )
                cv2.waitKey()
                cv2.destroyAllWindows()


def main():
    test(
        img_path="imagens/a01 - Original.bmp",
        output_directory="out a01/",
        output_name_template="a0",
        window_sizes=[(3, 3), (3, 13), (11, 1), (51, 21)],
        blur_algorithms=algorithms.items(),
        show_images=True,
    )
    test(
        img_path="imagens/b01 - Original.bmp",
        output_directory="out b01/",
        output_name_template="b0",
        window_sizes=[(7, 7), (11, 15)],
        blur_algorithms=algorithms.items(),
        show_images=True,
    )


if __name__ == "__main__":
    main()
