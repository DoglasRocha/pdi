import cv2, numpy as np


def binarize(img, threshold):
    """Binarização simples por limiarização.

    Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
                  canal independentemente.
                threshold: limiar.

    Valor de retorno: versão binarizada da img_in."""

    img = np.where(img < threshold, 0.0, 1.0)
    return img


# -------------------------------------------------------------------------------
def flood_fill(img, label, row, col, channel, dados_rotulo=None, updating=False):
    if dados_rotulo == None:
        dados_rotulo = {
            "label": label,
            "n_pixels": 1,
            "T": row,
            "L": col,
            "B": row,
            "R": col,
        }

    stack = [(row, col)]

    while len(stack) > 0:
        row, col = stack.pop()
        # vizinhança-4
        for neighbour_row in range(row - 1, row + 2):
            if 0 > neighbour_row or neighbour_row >= img.shape[0]:
                continue

            if img[neighbour_row][col][channel] == 1:
                img[neighbour_row][col][channel] = label

                dados_rotulo["n_pixels"] += 1

                if not updating:
                    dados_rotulo["T"] = min(neighbour_row, dados_rotulo["T"])
                    dados_rotulo["B"] = max(neighbour_row, dados_rotulo["B"])

                stack.append((neighbour_row, col))

        for neighbour_col in range(col - 1, col + 2):
            if 0 > neighbour_col or neighbour_col >= img.shape[1]:
                continue

            if img[row][neighbour_col][channel] == 1:
                img[row][neighbour_col][channel] = label

                dados_rotulo["n_pixels"] += 1

                if not updating:
                    dados_rotulo["L"] = min(neighbour_col, dados_rotulo["L"])
                    dados_rotulo["R"] = max(neighbour_col, dados_rotulo["R"])

                stack.append((row, neighbour_col))

    return dados_rotulo


def label(img, largura_min=0, altura_min=0, n_pixels_min=0):
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

    label = 2
    componentes = []
    tmp_img = img.copy()

    # inundação
    for row in range(len(tmp_img)):
        for col in range(len(tmp_img[row])):
            for channel in range(len(tmp_img[row][col])):
                if tmp_img[row][col][channel] != 1:
                    continue

                dados_componente = flood_fill(tmp_img, label, row, col, channel)
                if (
                    dados_componente["n_pixels"] < n_pixels_min
                    or dados_componente["R"] - dados_componente["L"] < largura_min
                    or dados_componente["B"] - dados_componente["T"] < altura_min
                ):
                    continue

                componentes.append(dados_componente)
                label += 1

    return componentes


def update_components(
    img: cv2.typing.MatLike,
    components: "list[dict]",
    largura_min=0,
    altura_min=0,
    n_pixels_min=0,
) -> None:
    tmp_img = img.copy()
    label = 2
    for i, c in enumerate(components):
        # inundação
        for row in range(c["T"], c["B"] + 1):
            for col in range(c["L"], c["R"] + 1):
                for channel in range(len(tmp_img[row, col])):
                    if tmp_img[row, col, channel] != 1:
                        continue

                    components[i]["n_pixels"] = 1
                    components[i] = flood_fill(
                        tmp_img, label, row, col, channel, components[i], True
                    )
                    if (
                        c["n_pixels"] < n_pixels_min
                        or c["R"] - c["L"] < largura_min
                        or c["B"] - c["T"] < altura_min
                    ):
                        continue

                    label += 1

    return components
