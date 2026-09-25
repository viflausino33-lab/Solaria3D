import numpy as np
import torch
from PIL import Image


def preparar_imagem(imagem, tamanho=256):

    if imagem is None:
        raise ValueError(
            "Nenhuma imagem fornecida."
        )

    if not isinstance(imagem, Image.Image):
        imagem = Image.fromarray(
            np.asarray(imagem)
        )

    imagem = imagem.convert("RGB")

    imagem = imagem.resize(
        (tamanho, tamanho),
        Image.Resampling.BILINEAR
    )

    imagem = np.asarray(
        imagem,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        imagem
    ).permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    return tensor
