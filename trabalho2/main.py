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
        w1 = int(w1)
        w2 = int(w2)
    except:
        print(
            "Por favor, informe o caminho para a imagem, "
            + "o algoritmo desejado, a largura da janela e a"
            + " altura da janela"
        )
        exit()

    image = open_image(img_path)
    result = blur.naive_algorithm(image, w1, w2)

    cv2.imshow(f"{img_path} original", image)
    cv2.imshow(f"{img_path} {algorithm} {w1}x{w2}", result)
    cv2.imwrite(
        f"out/{os.path.basename(img_path)} {algorithm} {w1}x{w2}.png", result * 255
    )
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
