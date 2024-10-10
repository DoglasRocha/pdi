# ===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import timeit
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = "arroz.bmp"

NEGATIVO = False
THRESHOLD = 0.775
ALTURA_MIN = 3
LARGURA_MIN = 3
N_PIXELS_MIN = 4

# INPUT_IMAGE = "documento-3mp.bmp"

# NEGATIVO = True
# THRESHOLD = 0.6
# ALTURA_MIN = 2
# LARGURA_MIN = 2
# N_PIXELS_MIN = 4

# ===============================================================================


def binariza(img, threshold):
    """Binarização simples por limiarização.

    Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
                  canal independentemente.
                threshold: limiar.

    Valor de retorno: versão binarizada da img_in."""

    img = np.where(img < threshold, 0.0, 1.0)
    return img


# -------------------------------------------------------------------------------


def inunda(img, label, row, col, channel, dados_rotulo=None):
    if dados_rotulo == None:
        dados_rotulo = {
            "label": label,
            "n_pixels": 1,
            "T": row,
            "L": col,
            "B": row,
            "R": col,
        }

    # vizinhança-8
    for neighbour_row in range(row - 1, row + 2):
        if 0 > neighbour_row or neighbour_row >= len(img):
            continue

        for neighbour_col in range(col - 1, col + 2):
            if 0 > neighbour_col or neighbour_col >= len(img[y]):
                continue

            if img[neighbour_row][neighbour_col][channel] == 1:
                img[neighbour_row][neighbour_col][channel] = label

                dados_rotulo["n_pixels"] += 1

                dados_rotulo["T"] = min(neighbour_row, dados_rotulo["T"])
                dados_rotulo["B"] = max(neighbour_row, dados_rotulo["B"])

                dados_rotulo["L"] = min(neighbour_col, dados_rotulo["L"])
                dados_rotulo["R"] = max(neighbour_col, dados_rotulo["R"])

                inunda(img, label, neighbour_row, neighbour_col, channel, dados_rotulo)

    return dados_rotulo


def rotula(img, largura_min, altura_min, n_pixels_min):
    """Rotulagem usando flood fill. Marca os objetos da imagem com os valores
    [0.1,0.2,etc].

    Parâmetros: img: imagem de entrada E saída.
                largura_min: descarta componentes com largura menor que esta.
                altura_min: descarta componentes com altura menor que esta.
                n_pixels_min: descarta componentes com menos pixels que isso.

    Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
    com os seguintes campos:

    'label': rótulo do componente.
    'n_pixels': número de pixels do componente.
    'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
    respectivamente: topo, esquerda, baixo e direita."""

    label = 0.1
    componentes = []
    # inundação
    for row in range(len(img)):
        for col in range(len(img[row])):
            for channel in range(len(img[row][col])):
                if img[row][col][channel] != 1:
                    continue

                dados_componente = inunda(img, label, row, col, channel)
                if (
                    dados_componente["n_pixels"] < n_pixels_min
                    or dados_componente["R"] - dados_componente["L"] < largura_min
                    or dados_componente["B"] - dados_componente["T"] < altura_min
                ):
                    continue

                componentes.append(dados_componente)
                label += 0.1

    return componentes


# ===============================================================================


def main():

    # Abre a imagem em escala de cinza.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Erro abrindo a imagem.\n")
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], 1))
    img = img.astype(np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza(img, THRESHOLD)
    cv2.imshow("01 - binarizada", img * 255)
    cv2.imwrite("01 - binarizada.png", img * 255)

    start_time = timeit.default_timer()
    componentes = rotula(img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len(componentes)
    print("Tempo: %f" % (timeit.default_timer() - start_time))
    print("%d componentes detectados." % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

    cv2.imshow("02 - out", img_out)
    cv2.imwrite("02 - out.png", img_out * 255)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

# ===============================================================================
