# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
import cv2, numpy as np
import blur
import os


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
    integral_img = blur.build_integral_image(img)
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
                cv2.imshow(
                    f"{output_name_template}{window_index + 2} - Borrada {w_size} - algoritmo {name}",
                    result,
                )
                cv2.imshow(f"opencv blur MENOS resultado", comparison)
                cv2.imshow(f"opencv blur MENOS resultado, normalizado", norm_comparison)
                cv2.waitKey()
                cv2.destroyAllWindows()


def main():
    test(
        img_path="imagens/a01 - Original.bmp",
        output_directory="out a01/",
        output_name_template="a0",
        window_sizes=[(3, 3), (3, 13), (11, 1), (51, 21)],
        blur_algorithms=blur.algorithms.items(),
        show_images=True,
    )
    test(
        img_path="imagens/b01 - Original.bmp",
        output_directory="out b01/",
        output_name_template="b0",
        window_sizes=[(7, 7), (11, 15)],
        blur_algorithms=blur.algorithms.items(),
        show_images=True,
    )


if __name__ == "__main__":
    main()
