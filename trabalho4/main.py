# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os

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

    stack = [(row, col)]

    while len(stack) > 0:
        row, col = stack.pop()
        # vizinhança-8
        for neighbour_row in range(row - 1, row + 2):
            if 0 > neighbour_row or neighbour_row >= len(img):
                continue

            for neighbour_col in range(col - 1, col + 2):
                if 0 > neighbour_col or neighbour_col >= len(img[neighbour_row]):
                    continue

                if img[neighbour_row][neighbour_col][channel] == 1:
                    img[neighbour_row][neighbour_col][channel] = label

                    dados_rotulo["n_pixels"] += 1

                    dados_rotulo["T"] = min(neighbour_row, dados_rotulo["T"])
                    dados_rotulo["B"] = max(neighbour_row, dados_rotulo["B"])

                    dados_rotulo["L"] = min(neighbour_col, dados_rotulo["L"])
                    dados_rotulo["R"] = max(neighbour_col, dados_rotulo["R"])

                    stack.append((neighbour_row, neighbour_col))

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

    label = 2
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
                label += 1

    return componentes

assert len(sys.argv) >= 2, "Por favor, insira como argumento para o script um caminho para uma imagem"
img_path = sys.argv[1]
assert os.path.exists(img_path), "Por favor, insira um caminho válido"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = img.reshape((img.shape[0], img.shape[1], 1))
img = img.astype(np.float32) / 255.0
img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

blurry = cv2.blur(img, (60, 60)).reshape((img.shape[0], img.shape[1], 1))

# normalizacao local
diff = img - blurry
# erodida = cv2.erode(diff, np.ones((3, 3)))
# erodida = cv2.erode(erodida, np.ones((3, 3)))

# # binariza
# binaria = erodida.copy()
binaria = diff.copy()
binaria = binariza(binaria, 0.15)

# erodida = cv2.erode(erodida, np.ones((3, 3)))

# rotula
componentes = rotula(binaria, 3, 3, 6)
n_componentes = len(componentes)
print("%d componentes detectados." % n_componentes)

# Mostra os objetos encontrados.
for c in componentes:
    cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

cv2.imshow("original", img)
# cv2.imshow("erodida", erodida)
# cv2.imshow("blur", blurry)
# cv2.imshow("diff", diff)
cv2.imshow("binarizada", binaria)
cv2.imshow("saida", img_out)

cv2.waitKey()
cv2.destroyAllWindows()