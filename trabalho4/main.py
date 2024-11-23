# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *


def calculate_metrics(components: "list[dict]") -> "tuple[float, float, float, float]":
    n_pixels_per_component = [component["n_pixels"] for component in components]
    if len(n_pixels_per_component) == 0:
        return 0, 0, 0, 0, 0

    mean = np.mean(n_pixels_per_component)
    std = np.std(n_pixels_per_component)
    min_ = np.min(n_pixels_per_component)
    max_ = np.max(n_pixels_per_component)
    median = np.median(n_pixels_per_component)

    return mean, std, min_, max_, median


def filter_abnormal_components(
    components: "list[dict]", threshold: float
) -> "list[dict]":
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
    mean, std, _, _, _ = calculate_metrics(components)

    normal_components = components.copy()
    abnormal_components = []
    small_components = []
    while std / mean > 0.2:  # and i < 100:
        abnormal_components.extend(
            filter_abnormal_components(normal_components, mean + std)
        )
        small_components.extend(filter_small_components(normal_components, mean - std))
        normal_components = filter_normal_components(
            normal_components, mean - std, mean + std
        )
        mean, std, _, _, _ = calculate_metrics(normal_components)

    return small_components, normal_components, abnormal_components


def count_rice(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    sharpen_img = sharp_image(img, kernel_size=(15, 15))
    binarized_img = normalize_locally_and_binarize_image(sharpen_img)
    non_noisy_img = supress_image_noise(
        binarized_img,
        np.ones((5, 5)),
    )

    # labeling
    components = label(non_noisy_img)

    mean, std, min_, max_, median = calculate_metrics(components)
    coef_variacao = std / mean
    c2 = lambda x: 1 if x <= 1 else -(x ** (1 / 10)) + 2
    constante_de_rocha = coef_variacao * c2(mean / median)
    print(
        f"{std=} {median=} {mean=} {min_=} {coef_variacao=} {median/mean=} {c2(mean / median)=}\n {constante_de_rocha=}"
    )
    n_rice = 0
    for c in components:
        hip_n_rice = round(c["n_pixels"] / (median * constante_de_rocha))
        n_rice += hip_n_rice

    print(f"{n_rice} componentes detectados.")

    for c in components:
        color = np.random.uniform(high=0.8, size=3)
        for row, col in c["positions"]:
            img_out[row, col] = color

    return img_out


assert (
    len(sys.argv) >= 2
), "Por favor, insira como argumento para o script um caminho para uma imagem"
img_path = sys.argv[1]
assert os.path.exists(img_path), "Por favor, insira um caminho válido"

img = open_image(img_path)
img_out = count_rice(img)

cv2.imshow("original", img)
cv2.imshow("saida", img_out)

cv2.waitKey()
cv2.destroyAllWindows()
