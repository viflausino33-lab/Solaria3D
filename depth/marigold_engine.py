from .depth_engine import DepthEngine, DepthResult


class MarigoldEngine(DepthEngine):
    """
    Adaptador da Solaria3D para o Marigold-V2.

    O carregamento real do modelo será habilitado
    quando estivermos em um ambiente com GPU compatível.
    """

    def __init__(self):
        super().__init__()
        self.model_name = "Marigold-V2"
        self.loaded = False

    def carregar_modelo(self):
        """
        Prepara o carregamento do Marigold-V2.
        """

        raise RuntimeError(
            "Marigold-V2 requer um ambiente com GPU CUDA "
            "compatível para inferência."
        )

    def analisar(self, imagem) -> DepthResult:
        """
        Executa a análise de profundidade com Marigold-V2.
        """

        if imagem is None:
            raise ValueError("Nenhuma imagem foi fornecida.")

        if not self.loaded:
            raise RuntimeError(
                "O Marigold-V2 ainda não foi carregado."
            )

        raise NotImplementedError(
            "A inferência do Marigold-V2 será conectada nesta etapa."
        )
