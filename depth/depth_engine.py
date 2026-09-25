from dataclasses import dataclass
from typing import Optional


@dataclass
class DepthResult:
    """
    Resultados produzidos pelo motor de profundidade.
    """

    depth: object
    normals: Optional[object] = None
    albedo: Optional[object] = None


class DepthEngine:
    """
    Motor de análise de profundidade da Solaria3D.

    O modelo de IA será conectado posteriormente.
    """

    def __init__(self):
        self.model = None

    def carregar_modelo(self):
        """
        Carrega o modelo de profundidade.
        """
        raise NotImplementedError(
            "O modelo de profundidade ainda não foi configurado."
        )

    def analisar(self, imagem) -> DepthResult:
        """
        Analisa uma imagem e retorna os resultados de profundidade.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        raise NotImplementedError(
            "A análise de profundidade ainda não foi configurada."
        )
