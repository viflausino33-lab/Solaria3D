from dataclasses import dataclass
from typing import Optional


@dataclass
class MarigoldResult:
    depth: Optional[object] = None
    normals: Optional[object] = None
    albedo: Optional[object] = None


class MarigoldEngine:
    """
    Interface da Solaria3D para o Marigold-V2.

    Nesta primeira versão, o modelo ainda não é carregado.
    A classe prepara a arquitetura para receber a implementação
    oficial posteriormente.
    """

    def __init__(self):
        self.model_loaded = False
        self.model = None

    def carregar_modelo(self):
        """
        Carrega os checkpoints do Marigold-V2.

        Será implementado quando tivermos o ambiente CUDA adequado.
        """

        raise RuntimeError(
            "Marigold-V2 ainda não está carregado. "
            "O ambiente CUDA será configurado posteriormente."
        )

    def analisar_depth(self, imagem):
        """
        Executa a estimativa de profundidade.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        raise RuntimeError(
            "Marigold-V2 Depth ainda não foi conectado."
        )

    def analisar_normals(self, imagem):
        """
        Executa a estimativa das normais da superfície.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        raise RuntimeError(
            "Marigold-V2 Normals ainda não foi conectado."
        )

    def analisar_albedo(self, imagem):
        """
        Executa a estimativa de albedo.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        raise RuntimeError(
            "Marigold-V2 Albedo ainda não foi conectado."
        )

    def analisar(self, imagem) -> MarigoldResult:
        """
        Executa todas as modalidades disponíveis.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        raise RuntimeError(
            "Marigold-V2 ainda não foi conectado."
        )
