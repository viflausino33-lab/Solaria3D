import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Bloco convolucional básico da SolariaDepth.

    Entrada:
        [B, canais, H, W]

    Saída:
        [B, canais_saida, H, W]
    """

    def __init__(self, canais_entrada, canais_saida):

        super().__init__()

        self.bloco = nn.Sequential(

            nn.Conv2d(
                canais_entrada,
                canais_saida,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                canais_saida
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                canais_saida,
                canais_saida,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                canais_saida
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.bloco(x)


class SolariaDepth(nn.Module):
    """
    Rede neural própria da Solaria3D
    para estimativa monocular de profundidade.

    Entrada:
        RGB [B, 3, H, W]

    Saída:
        Depth [B, 1, H, W]

    A saída é normalizada entre 0 e 1.
    """

    def __init__(self):

        super().__init__()

        # ====================================================
        # ENCODER
        # ====================================================

        self.enc1 = ConvBlock(
            3,
            32
        )

        self.enc2 = ConvBlock(
            32,
            64
        )

        self.enc3 = ConvBlock(
            64,
            128
        )

        self.enc4 = ConvBlock(
            128,
            256
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        self.bottleneck = ConvBlock(
            256,
            512
        )

        # ====================================================
        # DECODER
        # ====================================================

        self.up4 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )

        self.dec4 = ConvBlock(
            512,
            256
        )

        self.up3 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )

        self.dec3 = ConvBlock(
            256,
            128
        )

        self.up2 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )

        self.dec2 = ConvBlock(
            128,
            64
        )

        self.up1 = nn.ConvTranspose2d(
            64,
            32,
            kernel_size=2,
            stride=2
        )

        self.dec1 = ConvBlock(
            64,
            32
        )

        # ====================================================
        # SAÍDA
        # ====================================================

        self.output = nn.Sequential(

            nn.Conv2d(
                32,
                1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        # ====================================================
        # ENCODER
        # ====================================================

        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # ====================================================
        # BOTTLENECK
        # ====================================================

        b = self.bottleneck(
            self.pool(e4)
        )

        # ====================================================
        # DECODER
        # ====================================================

        d4 = self.up4(
            b
        )

        d4 = torch.cat(
            [d4, e4],
            dim=1
        )

        d4 = self.dec4(
            d4
        )

        d3 = self.up3(
            d4
        )

        d3 = torch.cat(
            [d3, e3],
            dim=1
        )

        d3 = self.dec3(
            d3
        )

        d2 = self.up2(
            d3
        )

        d2 = torch.cat(
            [d2, e2],
            dim=1
        )

        d2 = self.dec2(
            d2
        )

        d1 = self.up1(
            d2
        )

        d1 = torch.cat(
            [d1, e1],
            dim=1
        )

        d1 = self.dec1(
            d1
        )

        # ====================================================
        # DEPTH
        # ====================================================

        depth = self.output(
            d1
        )

        return depth
