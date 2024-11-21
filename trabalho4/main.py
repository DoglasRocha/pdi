# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *

KERNEL_LINHA = [
    [0, 0, 0, 0, 0], 
    [0, 0, 0, 0, 0], 
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]
KERNEL_COLUNA = [
    [0, 0, 0, 0, 0], 
    [0, 0, 1, 0, 0], 
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]
KERNEL_QUADRADO = [
    [0, 0, 0, 0, 0], 
    [0, 1, 0, 1, 0], 
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

def calculate_metrics(components: "list[dict]") -> "tuple[float, float, float, float]":
    n_pixels_per_component = [component["n_pixels"] for component in components]
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


def erode_abnormalities(
    img: cv2.typing.MatLike,
    components: "list[dict]",
    kernel: "np.ndarray",
) -> cv2.typing.MatLike:
    kernel = kernel.astype(np.uint8)
    for c in components:
        not_component = []
        top = c["T"] - kernel.shape[0] // 2
        bottom = c["B"] + kernel.shape[0] // 2
        left = c["L"] - kernel.shape[1] // 2
        right = c["R"] + kernel.shape[1] // 2

        # remove coisas que não são o componente
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                for channel in range(img.shape[2]):
                    if img[row, col, channel] == 1 and (row, col) not in c["positions"]:
                        not_component.append((row, col))
                        img[row, col, channel] = 0

        component_area = img[top:bottom, left:right, :]
        eroded_area = cv2.erode(component_area, kernel)
        eroded_area = reshape_image(eroded_area)

        img[top:bottom, left:right, :] = eroded_area

        # devolve coisas que não são o componente
        for row, col in not_component:
            for channel in range(img.shape[2]):
                img[row, col, channel] = 1

    return img


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


def remove_components_from_image(
    img: cv2.typing.MatLike, components: "list[dict]"
) -> cv2.typing.MatLike:
    for c in components:
        for row, col in c["positions"]:
            for channel in range(img.shape[2]):
                img[row, col, channel] = 0

    cv2.destroyAllWindows()
    return img


def re_add_components_into_image(
    img: cv2.typing.MatLike, components: "list[dict]"
) -> cv2.typing.MatLike:
    for c in components:
        for row, col in c["positions"]:
            for channel in range(img.shape[2]):
                img[row, col, channel] = 1

    return img


def treat_normal_components(
    img: cv2.typing.MatLike, components: "list[dict]"
) -> "tuple[cv2.typing.MatLike, list[dict]]":
    # se o centro do componente não está nas posições do componente, provavelmente é um arroz encostado no outro
    tmp_components = components.copy()

    for c in components:
        center = (c["B"] + c["T"]) // 2, (c["R"] + c["L"]) // 2

        if center not in c["positions"]:
            if c['B'] - c['T'] > (c['R'] - c['L']) * 1.75: 
                kernel = KERNEL_LINHA
                
            elif c['R'] - c['L'] > (c['B'] - c['T']) * 1.75:
                kernel = KERNEL_COLUNA
                
            else:
                kernel = KERNEL_QUADRADO
            while len(tmp_components) == len(components):
                img = erode_abnormalities(
                    img, 
                    [c], 
                    np.array(kernel)
                )
                tmp_components = label(img)

            components = tmp_components

    return img, components


def count_rice(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    sharpen_img = sharp_image(img, kernel_size=(15, 15))
    binarized_img = normalize_locally_and_binarize_image(sharpen_img)
    non_noisy_img = supress_image_noise(
        binarized_img, 
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0]
            ]
        )
    )

    # labeling
    components = label(non_noisy_img)
    small_components, normal_components, abnormal_components = classify_components(
        components
    )

    # componentes pequenos são confiáveis (dada a remoção de ruido)
    tmp_img = non_noisy_img.copy()
    final_components = small_components.copy()
    tmp_img = remove_components_from_image(tmp_img, small_components)

    # avalia componentes normais
    tmp_img = remove_components_from_image(tmp_img, abnormal_components)

    tmp_img, normal_components = treat_normal_components(tmp_img, normal_components)
    final_components.extend(normal_components)

    tmp_img = re_add_components_into_image(tmp_img, abnormal_components)
    tmp_img = remove_components_from_image(tmp_img, normal_components)

    # avalia componentes anormais
    normal_mean, normal_std, normal_min, _, normal_median = calculate_metrics(
        normal_components
    )

    # hipotese: numero max de arroz por blob = tamanho blob / (media ponderada da mediana + tamanho minimo)
    for c in abnormal_components:
        hip_n_rice = round(c["n_pixels"] / ((normal_median * 9 + normal_min) / 10))
        if c['B'] - c['T'] > (c['R'] - c['L']) * 1.75: 
            kernel = KERNEL_LINHA
            
        elif c['R'] - c['L'] > (c['B'] - c['T']) * 1.75:
            kernel = KERNEL_COLUNA
            
        else:
            kernel = KERNEL_QUADRADO

        # tenta chegar em numero de arrozes em numero de arrozes iteracoes
        n_components = -1
        i = 0
        while n_components != hip_n_rice or i < hip_n_rice:
            tmp_img = erode_abnormalities(
                tmp_img, 
                [c], 
                np.array(
                    kernel
                )
            )

            frame = tmp_img[c["T"] : c["B"], c["L"] : c["R"], :]
            tmp_components = label(frame)
            n_components = len(tmp_components)
            if n_components == hip_n_rice or n_components == 0:
                break

            i += 1

    abnormal_components = label(tmp_img)

    print(f"{len(final_components) + len(abnormal_components)} componentes detectados.")
    max_color = 0.8
    for c in normal_components:
        color = (np.random.uniform(high=max_color), np.random.uniform(high=max_color), np.random.uniform(high=max_color))
        for row, col in c["positions"]:
            img_out[row, col] = color
        # cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), color, 1)
    for c in abnormal_components:
        color = (np.random.uniform(high=max_color), np.random.uniform(high=max_color), np.random.uniform(high=max_color))
        for row, col in c["positions"]:
            img_out[row, col] = color
        # cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), color, 1)
    for c in small_components:
        color = (np.random.uniform(high=max_color), np.random.uniform(high=max_color), np.random.uniform(high=max_color))
        for row, col in c["positions"]:
            img_out[row, col] = color
        # cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), color, 1)

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
