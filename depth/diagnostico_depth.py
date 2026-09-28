import numpy as np
from PIL import Image

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine


IMAGE_PATH = "images.png"


print("========================================")
print("       DIAGNÓSTICO SOLARIA DEPTH")
print("========================================")


# ============================================================
# IMAGEM
# ============================================================

print()
print("Carregando imagem...")

imagem = Image.open(
    IMAGE_PATH
).convert("RGB")

print("Imagem original:", imagem.size)


# ============================================================
# SEGMENTAÇÃO
# ============================================================

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


# ============================================================
# ALPHA
# ============================================================

if objeto.mode == "RGBA":

    rgba = np.array(objeto)

    alpha = rgba[:, :, 3]

    pixels_validos = alpha > 20

    print()
    print("ALPHA")
    print("----------------------------------------")

    print("Min:", alpha.min())
    print("Max:", alpha.max())

    print(
        "Pixels objeto:",
        pixels_validos.sum()
    )

    print(
        "Pixels totais:",
        alpha.size
    )

    print(
        "Percentual objeto:",
        f"{100 * pixels_validos.mean():.2f}%"
    )

else:

    print()
    print("AVISO: objeto não está em RGBA.")


# ============================================================
# SOLARIA DEPTH
# ============================================================

print()
print("Carregando SolariaDepth...")

engine = SolariaDepthEngine()

print()
print("Gerando depth...")

depth = engine.analisar(
    imagem
)

depth = np.asarray(
    depth,
    dtype=np.float32
)


# ============================================================
# ESTATÍSTICAS GERAIS
# ============================================================

print()
print("DEPTH")
print("----------------------------------------")

print("Shape:", depth.shape)
print("dtype:", depth.dtype)

print("Min:", float(depth.min()))
print("Max:", float(depth.max()))
print("Média:", float(depth.mean()))
print("Mediana:", float(np.median(depth)))
print("Desvio padrão:", float(depth.std()))


# ============================================================
# DISTRIBUIÇÃO
# ============================================================

print()
print("DISTRIBUIÇÃO DA DEPTH")
print("----------------------------------------")

hist, bins = np.histogram(
    depth,
    bins=10,
    range=(0.0, 1.0)
)

for i in range(10):

    print(
        f"{bins[i]:.1f} - {bins[i + 1]:.1f}:",
        hist[i]
    )


# ============================================================
# DEPTH DENTRO DO OBJETO
# ============================================================

if objeto.mode == "RGBA":

    alpha = np.array(objeto)[:, :, 3]

    mask = alpha > 20

    mask_img = Image.fromarray(
        (mask * 255).astype(np.uint8)
    )

    mask_img = mask_img.resize(
        (
            depth.shape[1],
            depth.shape[0]
        ),
        Image.Resampling.NEAREST
    )

    mask = np.array(mask_img) > 0

    depth_objeto = depth[mask]

    print()
    print("DEPTH DENTRO DO OBJETO")
    print("----------------------------------------")

    print(
        "Pixels:",
        depth_objeto.size
    )

    if depth_objeto.size > 0:

        print(
            "Min:",
            float(depth_objeto.min())
        )

        print(
            "Max:",
            float(depth_objeto.max())
        )

        print(
            "Média:",
            float(depth_objeto.mean())
        )

        print(
            "Mediana:",
            float(np.median(depth_objeto))
        )

        print(
            "Desvio padrão:",
            float(depth_objeto.std())
        )


# ============================================================
# AMOSTRA DE LINHAS
# ============================================================

print()
print("AMOSTRA DA DEPTH")
print("----------------------------------------")

h, w = depth.shape

linhas = [
    h // 4,
    h // 2,
    (3 * h) // 4
]

for y in linhas:

    valores = depth[y, ::32]

    print(
        f"Linha {y}:"
    )

    print(
        np.round(
            valores,
            3
        )
    )


print()
print("========================================")
print("       DIAGNÓSTICO CONCLUÍDO")
print("========================================")
