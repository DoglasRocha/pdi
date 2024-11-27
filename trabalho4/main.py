# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# ATENÇÃO: ESTE É UM PROGRAMA DE LINHA DE COMANDO
# MODO DE USO: python main.py endereço_da_imagem

import cv2, numpy as np, sys, os
from segmentation import *


def main():
    assert (
        len(sys.argv) >= 2
    ), "\n\nPor favor, insira como argumento para o script um caminho para uma imagem.\n\tExemplo: python main.py 205.bmp"
    img_path = sys.argv[1]
    assert os.path.exists(img_path), "Por favor, insira um caminho válido"

    img = open_image(img_path)
    img_out, rice_count = count_rice(img)

    print(f"Contagem de arrozes na imagem {img_path}: {rice_count}")
    cv2.imshow("original", img)
    cv2.imshow("saida", img_out)

    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite(f"{os.path.basename(img_path)} -> {rice_count}.png", img_out * 255)


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


def supress_image_noise(
    img: cv2.typing.MatLike,
    kernel: "np.ndarray" = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
) -> cv2.typing.MatLike:
    kernel = kernel.astype(np.uint8)
    img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    img = reshape_image(img)
    return img


def normalize_locally(
    img: cv2.typing.MatLike, kernel_size: "tuple[int, int]" = (200, 200)
) -> cv2.typing.MatLike:
    blurry = cv2.blur(img, kernel_size)
    blurry = reshape_image(blurry)

    return img - blurry


def normalize_locally_and_binarize_image(
    img: cv2.typing.MatLike, threshold: float = 0.18
) -> cv2.typing.MatLike:
    normalized = normalize_locally(img)
    binarized = binarize(normalized, threshold)
    binarized = reshape_image(binarized)

    return binarized


def open_image(img_path: str) -> cv2.typing.MatLike:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = reshape_image(img)
    img = img.astype(np.float32) / 255.0

    return img


def reshape_image(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img = img.reshape((img.shape[0], img.shape[1], 1))
    return img


def calculate_metrics(components: "list[dict]") -> "dict":
    n_pixels_per_component = [component["n_pixels"] for component in components]
    if len(n_pixels_per_component) == 0:
        return 0, 0, 0, 0, 0

    return {
        "mean": np.mean(n_pixels_per_component),
        "std": np.std(n_pixels_per_component),
        "min": np.min(n_pixels_per_component),
        "max": np.max(n_pixels_per_component),
        "median": np.median(n_pixels_per_component),
        "cv": np.std(n_pixels_per_component) / np.mean(n_pixels_per_component),
    }


def filter_big_components(components: "list[dict]", threshold: float) -> "list[dict]":
    result = list(filter(lambda c: c["n_pixels"] > threshold, components))
    return result


def filter_normal_components(
    components: "list[dict]", low_threshold: float, high_threshold: float
) -> "list[dict]":
    result = list(
        filter(lambda c: low_threshold <= c["n_pixels"] <= high_threshold, components)
    )
    return result


def filter_small_components(components: "list[dict]", threshold: float) -> "list[dict]":
    result = list(filter(lambda c: c["n_pixels"] < threshold, components))
    return result


def classify_components(
    components: "list[dict]",
) -> "tuple[list[dict], list[dict], list[dict]]":
    metrics = calculate_metrics(components)

    normal_components = components.copy()
    big_components = []
    small_components = []
    # um coeficiente de variação de 20% garante que a amostra "normal" será relativamente homogênea,
    # ou seja, suas métricas como média e mediana serão mais confiáveis
    while metrics["cv"] > 0.2:
        big_components.extend(
            filter_big_components(normal_components, metrics["mean"] + metrics["std"])
        )
        small_components.extend(
            filter_small_components(normal_components, metrics["mean"] - metrics["std"])
        )
        normal_components = filter_normal_components(
            normal_components,
            metrics["mean"] - metrics["std"],
            metrics["mean"] + metrics["std"],
        )
        metrics = calculate_metrics(normal_components)

    return small_components, normal_components, big_components


def count_rice(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    binarized_img = normalize_locally_and_binarize_image(img)
    non_noisy_img = supress_image_noise(
        binarized_img,
        np.ones((5, 5)),
    )

    # labeling
    components = label(non_noisy_img)

    # small -> quebrados e pequenos, normal -> arrozes unicos (com poucas exceções)
    # big -> grandes componentes
    small_components, normal_components, big_components = classify_components(
        components
    )

    # metricas dos componentes normais
    normal_metrics = calculate_metrics(normal_components)

    # arrozes pequenos são confiáveis se o tratamento de ruído foi bem sucedido
    for c in small_components:
        c["rice_count"] = 1
    rice_count = len(small_components)

    # hehehehehe
    # o parametro de rocha serve para dar um leve shift pra esquerda na mediana,
    # para melhorar a contagem de arrozes quebrados em componentes
    parametro_de_rocha = normal_metrics["std"] * normal_metrics["cv"]

    for c in normal_components + big_components:
        c["rice_count"] = round(
            c["n_pixels"]
            / (
                normal_metrics["median"] - parametro_de_rocha
            )  # métricas dos componentes "normais" (tamanho médio) são mais confiáveis
            # para o cálculo de arrozes por componente
        )
        rice_count += c["rice_count"]

    components_and_colors = (
        (small_components, (0, 1, 0)),
        (normal_components, (1, 0, 0)),
        (big_components, (0, 0, 1)),
    )
    for components_, color in components_and_colors:
        for c in components_:
            for row, col in c["positions"]:
                img_out[row, col] = color
            img_out = cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), color)
            # outline
            img_out = cv2.putText(
                img_out,
                str(c["rice_count"]),
                (c["L"], c["T"]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 0),
                2,
            )
            # texto
            img_out = cv2.putText(
                img_out,
                str(c["rice_count"]),
                (c["L"], c["T"]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                color,
            )

    return img_out, rice_count


if __name__ == "__main__":
    main()
