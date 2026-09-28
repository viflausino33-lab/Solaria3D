import os

from PIL import Image

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine
from depth.pointcloud import (
    depth_to_pointcloud,
    save_pointcloud_ply
)


IMAGE_PATH = "images.png"
OUTPUT_PATH = "models/solaria_depth_v1.ply"


print("========================================")
print("       TESTE DE NUVEM DE PONTOS")
print("========================================")


print()
print("Carregando imagem...")

imagem = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Imagem:",
    imagem.size
)


print()
print("Segmentando objeto...")

objeto = remover_fundo(
    imagem
)

print(
    "Objeto:",
    objeto.size,
    objeto.mode
)


print()
print("Carregando SolariaDepth...")

engine = SolariaDepthEngine()


print()
print("Gerando profundidade...")

depth = engine.analisar(
    objeto
)

print(
    "Depth:",
    depth.shape
)


print()
print("Convertendo para nuvem de pontos...")

points, colors = depth_to_pointcloud(
    depth,
    image=objeto,
    mask=objeto,
    stride=2
)


print(
    "Pontos:",
    len(points)
)

print(
    "Formato:",
    points.shape
)

print(
    "Cores:",
    colors.shape
)


os.makedirs(
    "models",
    exist_ok=True
)


print()
print("Salvando PLY...")

save_pointcloud_ply(
    OUTPUT_PATH,
    points,
    colors
)


print()
print(
    "Arquivo:",
    OUTPUT_PATH
)

print(
    "Tamanho:",
    os.path.getsize(
        OUTPUT_PATH
    ),
    "bytes"
)

print()
print("========================================")
print("TESTE CONCLUÍDO COM SUCESSO")
print("========================================")
