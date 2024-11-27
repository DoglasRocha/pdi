# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# ATENÇÃO: ESTE É UM PROGRAMA DE LINHA DE COMANDO
# MODO DE USO: python main.py endereço_da_imagem

import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *


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
