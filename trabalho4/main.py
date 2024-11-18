# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *
from time import sleep


def calculate_metrics(components: "list[dict]") -> "tuple[float, float, float, float]":
    n_pixels_per_component = [component["n_pixels"] for component in components]
    mean = np.mean(n_pixels_per_component)
    std = np.std(n_pixels_per_component)
    min_ = np.min(n_pixels_per_component)
    max_ = np.max(n_pixels_per_component)

    return mean, std, min_, max_


def filter_abnormal_components(components: "list[dict]", threshold: float) -> "list[dict]":
    result = list(filter(lambda c: c['n_pixels'] > threshold, components))
    return result


def filter_normal_components(components: "list[dict]", low_threshold: float, high_threshold: float) -> "list[dict]":
    result = list(filter(lambda c: low_threshold <= c['n_pixels'] <= high_threshold, components))
    return result


def filter_small_components(components: "list[dict]", threshold: float) -> "list[dict]":
    result = list(filter(lambda c: c['n_pixels'] < threshold, components))
    return result

def erode_abnormalities(
    img: cv2.typing.MatLike,
    components: "list[dict]",
    threshold: float,
    kernel: "tuple[int, int]",
) -> cv2.typing.MatLike:

    for c in components:
        if c["n_pixels"] <= threshold:
            continue

        component_area = img[c["T"] - 1 : c["B"] + 1, c["L"] - 1 : c["R"] + 1, :]
        eroded_area = cv2.erode(component_area, np.ones(kernel))
        eroded_area = reshape_image(eroded_area)
        img[c["T"] - 1 : c["B"] + 1, c["L"] - 1 : c["R"] + 1, :] = eroded_area

    return img


def count_rice(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    binarized_img = normalize_locally_and_binarize_image(img)
    non_noisy_img = supress_image_noise(binarized_img, (5, 5))

    # labeling
    components = label(non_noisy_img)
    mean, std, min_, max_ = calculate_metrics(components)

    normal_components = components.copy()
    abnormal_components = []
    small_components = []
    i = 0
    while std / mean > 0.2:# and i < 100:
        abnormal_components.extend(filter_abnormal_components(normal_components, mean + std))
        small_components.extend(filter_small_components(normal_components, mean - std))
        normal_components = filter_normal_components(normal_components, mean - std, mean + std)
        mean, std, min_, max_ = calculate_metrics(normal_components)
        #i += 1

    final_binarized = non_noisy_img
    # cv2.imshow("hello", final_binarized)
    # cv2.waitKey(500)
    # if std / mean > 0.2:
    #     img_with_some_erosions = non_noisy_img.copy()
    #     new_components = components.copy()
    #     threshold = 0 * std + mean
    #     while max_ > threshold:
    #         img_with_some_erosions = erode_abnormalities(
    #             img_with_some_erosions, new_components, threshold, (3, 3)
    #         )
    #         img_with_some_erosions = supress_image_noise(img_with_some_erosions, (3, 3))
    #         components = update_components(img_with_some_erosions, components)
    #         new_components = label(img_with_some_erosions)
    #         _, _, _, max_ = calculate_metrics(components)

    #         cv2.imshow("hello", img_with_some_erosions)
    #         cv2.waitKey(500)

    #     cv2.destroyAllWindows()
    #     sleep(0.5)
    #     final_binarized = img_with_some_erosions

    # cv2.imshow("hello", final_binarized)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    components = label(final_binarized)
    print(f"{len(components)} componentes detectados.")
    for c in normal_components:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1), 2)
    for c in abnormal_components:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 1, 0), 2)
    for c in small_components:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (1, 0, 0), 2)

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
