# ===============================================================================
# Autor: Doglas Franco Maurer da Rocha
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

# processo: normalizacao local -> binarizacao -> segmentacao -> ver no que dá
import cv2, numpy as np, sys, os
from segmentation import *
from img_utils import *
from operations import *

def count_rice(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    sharped_image = sharp_image(img)
    # cv2.imshow("sharp", sharped_image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    binarized_img = normalize_locally_and_binarize_image(sharped_image)
    non_noisy_img = supress_image_noise(binarized_img, (5,5))

    # labeling
    components = label(non_noisy_img.copy(), 0, 0, 0)
    
    n_pixels_per_component = [component['n_pixels'] for component in components]
    mean = np.mean(n_pixels_per_component)
    std = np.std(n_pixels_per_component)

    print(f"{mean=} {std=} {std/mean=}")

    if (std / mean > 0.2):
        eroded = cv2.erode(non_noisy_img, np.ones((3, 3)))
        eroded = reshape_image(eroded)

        cv2.imshow("eroded", eroded)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(non_noisy_img.mean())
        print(eroded.mean())
        components = label(eroded.copy(), 0, 0, 0)
        n_pixels_per_component = [component['n_pixels'] for component in components]
        mean = np.mean(n_pixels_per_component)
        std = np.std(n_pixels_per_component)

        print(f"{mean=} {std=} {std/mean=}")

    print(f"{len(components)} componentes detectados.")
    for c in components:
        cv2.rectangle(img_out, (c["L"], c["T"]), (c["R"], c["B"]), (0, 0, 1))

    return img_out

assert len(sys.argv) >= 2, "Por favor, insira como argumento para o script um caminho para uma imagem"
img_path = sys.argv[1]
assert os.path.exists(img_path), "Por favor, insira um caminho válido"

img = open_image(img_path)
img_out = count_rice(img)

cv2.imshow("original", img)
cv2.imshow("saida", img_out)

cv2.waitKey()
cv2.destroyAllWindows()