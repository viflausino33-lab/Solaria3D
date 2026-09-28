import os

import numpy as np
import torch
from PIL import Image

from depth.model import SolariaDepth
from depth.preprocessing import preparar_imagem


class SolariaDepthEngine:

    def __init__(
        self,
        model_path="models/solaria_depth_v1.pth"
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Modelo não encontrado: {self.model_path}"
            )

        print("========================================")
        print("      SOLARIA DEPTH ENGINE")
        print("========================================")

        print("Dispositivo:", self.device)
        print("Modelo:", self.model_path)

        # ----------------------------------------
        # CRIA MODELO
        # ----------------------------------------

        self.model = SolariaDepth()

        # ----------------------------------------
        # CARREGA PESOS
        # ----------------------------------------

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        # ----------------------------------------
        # MODO INFERÊNCIA
        # ----------------------------------------

        self.model.to(
            self.device
        )

        self.model.eval()

        print("Modelo carregado com sucesso.")

    def analisar(self, imagem):

        if imagem is None:
            raise ValueError(
                "Nenhuma imagem foi fornecida."
            )

        if not isinstance(imagem, Image.Image):
            imagem = Image.fromarray(
                np.asarray(imagem)
            )

        imagem = imagem.convert("RGB")

        # ----------------------------------------
        # PREPARAÇÃO
        # ----------------------------------------

        entrada = preparar_imagem(
            imagem
        )

        entrada = entrada.to(
            self.device
        )

        # ----------------------------------------
        # INFERÊNCIA
        # ----------------------------------------

        with torch.no_grad():

            depth = self.model(
                entrada
            )

        # ----------------------------------------
        # REMOVE DIMENSÕES EXTRAS
        # ----------------------------------------

        depth = depth.squeeze()

        # ----------------------------------------
        # CPU / NUMPY
        # ----------------------------------------

        depth = depth.cpu().numpy()

        # ----------------------------------------
        # NORMALIZAÇÃO
        # ----------------------------------------

        minimo = depth.min()
        maximo = depth.max()

        if maximo > minimo:

            depth = (
                depth - minimo
            ) / (
                maximo - minimo
            )

        else:

            depth = np.zeros_like(
                depth,
                dtype=np.float32
            )

        return depth.astype(
            np.float32
        )
