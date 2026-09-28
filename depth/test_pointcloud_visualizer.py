from PIL import Image

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine
from depth.pointcloud import depth_to_pointcloud
from depth.pointcloud_visualizer import pointcloud_to_figure


IMAGE_PATH = "images.png"


print("========================================")
print("   TESTE VISUALIZAÇÃO DA NUVEM 3D")
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
print("Segmentando...")

objeto = remover_fundo(
    imagem
)

print(
    "Objeto:",
    objeto.mode,
    objeto.size
)


print()
print("Gerando profundidade...")

engine = SolariaDepthEngine()

depth = engine.analisar(
    objeto
)

print(
    "Depth:",
    depth.shape
)


print()
print("Gerando pontos...")

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


print()
print("Criando figura 3D...")

figura = pointcloud_to_figure(
    points,
    colors
)


figura.write_html(
    "models/solaria_pointcloud_test.html"
)


print()
print("Arquivo criado:")
print("models/solaria_pointcloud_test.html")


print()
print("========================================")
print("TESTE CONCLUÍDO")
print("========================================")
