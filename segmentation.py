from rembg import remove, new_session
from PIL import Image


# Carrega o modelo U²-Net uma única vez
session = new_session("u2net")


def remover_fundo(imagem):
    """
    Recebe uma imagem PIL e retorna a imagem
    com o fundo removido.
    """

    if imagem is None:
        return None

    resultado = remove(imagem, session=session)

    return resultado
