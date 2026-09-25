import numpy as np
from PIL import Image


def depth_to_image(depth):
    """
    Converte uma matriz de profundidade para uma imagem
    em tons de cinza.
    """

    if depth is None:
        return None

    depth = np.asarray(depth, dtype=np.float32)

    minimo = depth.min()
    maximo = depth.max()

    if maximo > minimo:
        depth = (depth - minimo) / (maximo - minimo)
    else:
        depth = np.zeros_like(depth)

    depth = (depth * 255).astype(np.uint8)

    return Image.fromarray(depth, mode="L")
