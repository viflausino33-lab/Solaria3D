import numpy as np
from PIL import Image

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine


IMAGE_PATH = "images.png"


print("========================================")
print("   DIAGNÓSTICO VISUAL SOLARIA DEPTH")
print("========================================")


# ============================================================
# IMAGEM
# ============================================================

print()
print("Carregando imagem...")

imagem = Image.open(
    IMAGE_PATH
).convert("RGB")

print(
    "Imagem:",
    imagem.size
)


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


rgba = np.array(
    objeto
)

alpha = rgba[:, :, 3]

mask = alpha > 20


# ============================================================
# DEPTH
# ============================================================

print()
print("Gerando depth...")

engine = SolariaDepthEngine()

depth = engine.analisar(
    imagem
)

depth = np.asarray(
    depth,
    dtype=np.float32
)


print(
    "Depth:",
    depth.shape
)


# ============================================================
# MÁSCARA PARA O TAMANHO DA DEPTH
# ============================================================

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

mask_depth = (
    np.array(mask_img) > 20
)


# ============================================================
# DEPTH SOMENTE NO OBJETO
# ============================================================

depth_objeto = depth.copy()

depth_objeto[
    ~mask_depth
] = 0


# ============================================================
# NORMALIZAÇÃO
# ============================================================

def normalizar(array):

    minimo = array.min()

    maximo = array.max()

    if maximo - minimo < 1e-8:

        return np.zeros_like(
            array,
            dtype=np.uint8
        )

    resultado = (
        (array - minimo)
        /
        (maximo - minimo)
        *
        255
    )

    return resultado.astype(
        np.uint8
    )


# ============================================================
# DEPTH COMPLETA
# ============================================================

depth_img = Image.fromarray(
    normalizar(depth)
)

depth_img.save(
    "models/diagnostico_depth.png"
)


# ============================================================
# MÁSCARA
# ============================================================

mask_output = Image.fromarray(
    (
        mask_depth.astype(
            np.uint8
        )
        *
        255
    )
)

mask_output.save(
    "models/diagnostico_mascara.png"
)


# ============================================================
# DEPTH SOMENTE NO OBJETO
# ============================================================

depth_objeto_img = Image.fromarray(
    normalizar(depth_objeto)
)

depth_objeto_img.save(
    "models/diagnostico_depth_objeto.png"
)


# ============================================================
# IMAGEM ORIGINAL REDUZIDA
# ============================================================

imagem_small = imagem.resize(
    (
        depth.shape[1],
        depth.shape[0]
    )
)

imagem_small.save(
    "models/diagnostico_original.png"
)


# ============================================================
# ESTATÍSTICAS
# ============================================================

valores_objeto = depth[
    mask_depth
]

print()
print("========================================")
print("ESTATÍSTICAS")
print("========================================")

print(
    "Pixels do objeto:",
    valores_objeto.size
)

print(
    "Depth objeto min:",
    float(valores_objeto.min())
)

print(
    "Depth objeto max:",
    float(valores_objeto.max())
)

print(
    "Depth objeto média:",
    float(valores_objeto.mean())
)

print(
    "Depth objeto mediana:",
    float(np.median(valores_objeto))
)

print(
    "Depth objeto desvio:",
    float(valores_objeto.std())
)


# ============================================================
# ARQUIVOS
# ============================================================

print()
print("Arquivos criados:")

print(
    "models/diagnostico_original.png"
)

print(
    "models/diagnostico_mascara.png"
)

print(
    "models/diagnostico_depth.png"
)

print(
    "models/diagnostico_depth_objeto.png"
)


print()
print("========================================")
print("       DIAGNÓSTICO CONCLUÍDO")
print("========================================")
