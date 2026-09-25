import numpy as np
from PIL import Image

from depth_engine import DepthEngine, DepthResult


class TestDepthEngine(DepthEngine):
    """
    Motor temporário usado apenas para validar
    a estrutura do pipeline.
    """

    def analisar(self, imagem) -> DepthResult:
        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        largura, altura = imagem.size

        # Cria um gradiente artificial de profundidade.
        depth = np.linspace(
            0,
            1,
            largura * altura,
            dtype=np.float32
        ).reshape(altura, largura)

        return DepthResult(
            depth=depth
        )


imagem = Image.new("RGB", (256, 256))

engine = TestDepthEngine()

resultado = engine.analisar(imagem)

print("DepthEngine executado com sucesso")
print("Formato:", resultado.depth.shape)
print("Tipo:", resultado.depth.dtype)
print("Mínimo:", resultado.depth.min())
print("Máximo:", resultado.depth.max())
print("Normals:", resultado.normals)
print("Albedo:", resultado.albedo)
