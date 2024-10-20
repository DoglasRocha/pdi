# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================
from sys import argv
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


def main():
    try:
        _, img_path, algorithm, w1, w2 = argv
        window_width = int(w1)
        window_height = int(w2)
    except:
        print(
            "Por favor, informe o caminho para a imagem, "
            + "o algoritmo desejado, a largura e a"
            + " altura da janela"
        )
        exit()

    image = open_image(img_path)
    if algorithm == "teste":
        for algorithm_ in blur.algorithms.keys():
            result = blur.algorithms[algorithm_](image, window_width, window_height)
            cv2.imshow(f"{img_path} original", image)
            cv2.imshow(
                f"{img_path} {algorithm_} {window_width}x{window_height}", result
            )
            cv2.imwrite(
                f"out/{os.path.basename(img_path)} {algorithm_} {window_width}x{window_height}.png",
                result * 255,
            )
    else:
        result = blur.algorithms[algorithm](image, window_width, window_height)
        cv2.imshow(f"{img_path} original", image)
        cv2.imshow(f"{img_path} {algorithm} {window_width}x{window_height}", result)
        cv2.imwrite(
            f"out/{os.path.basename(img_path)} {algorithm} {window_width}x{window_height}.png",
            result * 255,
        )
    cv2.waitKey()
    cv2.destroyAllWindows()


def test():
    PATH_IMG_1 = "imagens/a01 - Original.bmp"
    PATH_IMG_2 = "imagens/b01 - Original.bmp"

    # teste imagem 1, todos os tamanhos de janela
    print("======================== TESTE IMAGEM 1 ========================")
    window_sizes = [(3, 3), (3, 13), (11, 1), (51, 21)]
    img = open_image(PATH_IMG_1)
    img_integral = blur.build_integral_image(img)
    for index, w_size in enumerate(window_sizes):
        print(f"======== TAMANHO DE JANELA: {w_size} ========")
        img_ingenua = blur.blur_naive_algorithm(img, *w_size)
        img_separavel = blur.blur_separable_filter(img, *w_size)
        img_c_integral = blur.blur_integral_image(img, *w_size, img_integral)

        cv2.imwrite(
            f"out/a0{index + 2} - Borrada {w_size} algoritmo ingênuo.png", img_ingenua * 255
        )
        cv2.imwrite(
            f"out/a0{index + 2} - Borrada {w_size} algoritmo separável.png", img_separavel * 255
        )
        cv2.imwrite(
            f"out/a0{index + 2} - Borrada {w_size} algoritmo com imagem integral.png",
            img_c_integral * 255,
        )

    # teste imagem 2, todos os tamanhos de janela
    print("\n\n======================== TESTE IMAGEM 2 ========================")
    window_sizes = [(7, 7), (11, 15)]
    img = open_image(PATH_IMG_2)
    img_integral = blur.build_integral_image(img)
    for index, w_size in enumerate(window_sizes):
        print(f"======== TAMANHO DE JANELA: {w_size} ========")
        img_ingenua = blur.blur_naive_algorithm(img, *w_size)
        img_separavel = blur.blur_separable_filter(img, *w_size)
        img_c_integral = blur.blur_integral_image(img, *w_size, img_integral)

        cv2.imwrite(
            f"out/b0{index + 2} - Borrada {w_size} algoritmo ingênuo.png", img_ingenua * 255
        )
        cv2.imwrite(
            f"out/b0{index + 2} - Borrada {w_size} algoritmo separável.png", img_separavel * 255
        )
        cv2.imwrite(
            f"out/b0{index + 2} - Borrada {w_size} algoritmo com imagem integral.png",
            img_c_integral * 255,
        )


if __name__ == "__main__":
    test()
    # main()
