# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# ATENÇÃO: ESTE É UM PROGRAMA DE LINHA DE COMANDO
# MODO DE USO: python main.py endereço_da_imagem_com_verde endereço_da_imagem_a_ser_colada
import sys, os, cv2, numpy as np


def main() -> None:
    assert (
        len(sys.argv) >= 3
    ), "\n\nPor favor, insira como argumento para o script um caminho para as imagens.\n\tExemplo: python main.py 0.bmp img.png"
    img_path = sys.argv[1]
    stamp_img_path = sys.argv[2]
    assert os.path.exists(
        img_path
    ), "Por favor, insira um caminho válido para a imagem base"
    assert os.path.exists(
        stamp_img_path
    ), 'Por favor, insira um caminho válido para a imagem "carimbo"'

    img = cv2.imread(img_path).astype(np.float32) / 255
    stamp_img = cv2.imread(stamp_img_path).astype(np.float32) / 255

    stamp_img = cv2.resize(stamp_img, img.shape[::-1][1:])

    mask = np.ones(img.shape[:-1])

    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            b, g, r = img[row, col]
            if g > b and g > r and g * 0.666 > (b + r) / 2:
                # print(b, g, r)
                mask[row, col] = 1 - g

    mask = np.where(mask > 0.85, 1.0, mask)
    mask = cv2.normalize(mask, None, 0.0, 1.0, cv2.NORM_MINMAX)
    mask = np.round(mask, 3)

    mask_and_img = np.zeros(img.shape)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            b, g, r = img[row, col]
            if mask[row, col] == 1:
                mask_and_img[row, col] = mask[row, col] * np.array([b, g, r])
            else:
                mask_and_img[row, col] = mask[row, col] * np.array([b, 0, r])

    stamped_img = np.zeros(img.shape)
    for row in range(stamped_img.shape[0]):
        for col in range(stamped_img.shape[1]):
            if mask[row, col] == 0:
                stamped_img[row, col] = stamp_img[row, col]
            elif mask[row, col] == 1:
                stamped_img[row, col] = img[row, col]
            else:
                o_b, _, o_r = img[row, col]
                _, g, _ = stamp_img[row, col]
                g *= mask[row, col]
                stamped_img[row, col] = (1 - mask[row, col]) * stamp_img[
                    row, col
                ] + mask[row, col] * np.array([o_b, 0, o_r])

    cv2.imshow("original", img)
    cv2.imshow("mask", mask)
    cv2.imshow("mask and img", mask_and_img)
    cv2.imshow("result", stamped_img)
    # cv2.imwrite("aaaa.png", mask * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
