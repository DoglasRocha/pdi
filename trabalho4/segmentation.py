import cv2, numpy as np


def binarize(img, threshold):
    """Binarização simples por limiarização.

    Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
                  canal independentemente.
                threshold: limiar.

    Valor de retorno: versão binarizada da img_in."""

    img = np.where(img < threshold, 0.0, 1.0)
    return img


def inner_flood_fill(
    img,
    label,
    row,
    col,
    channel,
    changing_element,
    stack,
    dados_rotulo,
    lower_side,
    bigger_side,
    updating=False,
):
    if 0 > row or row >= img.shape[0] or 0 > col or col >= img.shape[1]:
        return

    if img[row][col][channel] == 1:
        img[row][col][channel] = label
        dados_rotulo["positions"].append((row, col))

        dados_rotulo["n_pixels"] += 1

        if not updating:
            dados_rotulo[lower_side] = min(changing_element, dados_rotulo[lower_side])
            dados_rotulo[bigger_side] = max(changing_element, dados_rotulo[bigger_side])

        stack.append((row, col))


# -------------------------------------------------------------------------------
def flood_fill(img, label, row, col, channel, dados_rotulo=None, updating=False):
    if dados_rotulo == None:
        dados_rotulo = {
            "label": label,
            "n_pixels": 0,
            "T": row,
            "L": col,
            "B": row,
            "R": col,
            "positions": [],
        }

    stack = [(row, col)]

    while len(stack) > 0:
        row, col = stack.pop()
        # vizinhança-4
        for neighbour_row in range(row - 1, row + 2):
            inner_flood_fill(
                img=img,
                label=label,
                row=neighbour_row,
                col=col,
                channel=channel,
                changing_element=neighbour_row,
                stack=stack,
                dados_rotulo=dados_rotulo,
                lower_side="T",
                bigger_side="B",
                updating=updating,
            )

        for neighbour_col in range(col - 1, col + 2):
            inner_flood_fill(
                img=img,
                label=label,
                row=row,
                col=neighbour_col,
                channel=channel,
                changing_element=neighbour_col,
                stack=stack,
                dados_rotulo=dados_rotulo,
                lower_side="L",
                bigger_side="R",
                updating=updating,
            )

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
                    label += 1
                    continue

                componentes.append(dados_componente)
                label += 1

    return componentes
