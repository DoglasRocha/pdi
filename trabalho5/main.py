# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# ATENÇÃO: ESTE É UM PROGRAMA DE LINHA DE COMANDO
# MODO DE USO: python main.py endereço_da_imagem_com_verde endereço_da_imagem_a_ser_colada
import cv2.typing
import sys, os, cv2, numpy as np


def main() -> None:
    assert (
        len(sys.argv) >= 3
    ), "\n\nPor favor, insira como argumento para o script um caminho para as imagens.\n\tExemplo: python main.py 0.bmp img.png"
    img_path = sys.argv[1]
    background_img_path = sys.argv[2]
    assert os.path.exists(
        img_path
    ), "Por favor, insira um caminho válido para a imagem base"
    assert os.path.exists(
        background_img_path
    ), "Por favor, insira um caminho válido para a que representa o fundo"

    img = read_image(img_path)
    original_img = img.copy()
    background_img = read_image(background_img_path)
    background_img = cv2.resize(background_img, img.shape[::-1][1:])

    mask = create_mask(img)
    img_without_background = create_img_without_background(img, mask)

    cv2.imshow("mask", mask)
    cv2.imshow("foreground", img_without_background)
    cv2.waitKey()
    cv2.destroyAllWindows()

    new_img = create_image_with_new_background(img, mask, background_img)

    cv2.imshow("original", original_img)
    cv2.imshow("result", new_img)
    cv2.imwrite("result.png", new_img * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


def read_image(img_path: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path).astype(np.float32) / 255
    return img


def create_mask(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    mask = np.ones(img.shape[:-1])

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            b, g, r = img[row, col]
            # se g > r e g > b e
            # 72,5% do g é maior que a média de b e r, então é bem verde
            if (g >= r and g >= b) and (g * 0.725 > (b + r) / 2):
                mask[row, col] = 1 - g

    mask = np.where(mask > 0.85, 1.0, mask)
    mask = cv2.normalize(mask, None, 0.0, 1.0, cv2.NORM_MINMAX)
    mask = np.round(mask, 3)  # em uma imagem específica (5.bmp), o valor máximo que a
    # normalização resultava era 0.9999 e isso quebrava todo o resto do algoritmo

    return mask


def create_img_without_background(
    img: cv2.typing.MatLike, mask: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    mask_and_img = np.zeros(img.shape)

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if mask[row, col] == 1:
                mask_and_img[row, col] = img[row, col]
            elif mask[row, col] == 0:
                mask_and_img[row, col] = 0
            else:
                img[row, col][1] = 0
                mask_and_img[row, col] = mask[row, col] * img[row, col]

    return mask_and_img


def create_image_with_new_background(
    img: cv2.typing.MatLike, mask: cv2.typing.MatLike, new_bg: cv2.typing.MatLike
) -> cv2.typing.MatLike:
    new_img = np.zeros(img.shape)

    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            # definitivamente fundo
            if mask[row, col] == 0:
                new_img[row, col] = new_bg[row, col]
            # definitvamente imagem original
            elif mask[row, col] == 1:
                new_img[row, col] = img[row, col]
            # definivamente nem fundo nem imagem original, provavelmente borda ou sombra
            else:
                # remove o verde
                # img[row, col][1] = 0
                new_img[row, col] = (1 - mask[row, col]) * new_bg[row, col] + mask[
                    row, col
                ] * img[row, col]

    return new_img


if __name__ == "__main__":
    main()
