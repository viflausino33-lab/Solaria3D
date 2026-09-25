import numpy as np
from PIL import Image

from .depth_engine import DepthEngine, DepthResult


class TestDepthEngine(DepthEngine):

    def analisar(self, imagem) -> DepthResult:
        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        largura, altura = imagem.size

        # Gera um mapa de profundidade artificial
        # apenas para testar o pipeline.
        depth = np.linspace(
            0,
            1,
            largura * altura,
            dtype=np.float32
        ).reshape(altura, largura)

        return DepthResult(
            depth=depth
        )


if __name__ == "__main__":

    # A imagem está na raiz do projeto.
    imagem = Image.open("images.webp").convert("RGB")

    engine = TestDepthEngine()

    resultado = engine.analisar(imagem)

    print("Depth Engine executado com sucesso")
    print("Formato:", resultado.depth.shape)
    print("Tipo:", resultado.depth.dtype)
    print("Mínimo:", resultado.depth.min())
    print("Máximo:", resultado.depth.max())
