import numpy as np
from PIL import Image


def preparar_depth(depth, tamanho):
    """
    Redimensiona o mapa de profundidade para o tamanho
    da imagem original.
    """

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    if depth.ndim != 2:
        depth = np.squeeze(depth)

    depth = np.clip(
        depth,
        0.0,
        1.0
    )

    imagem_depth = Image.fromarray(
        (depth * 255).astype(np.uint8),
        mode="L"
    )

    imagem_depth = imagem_depth.resize(
        tamanho,
        Image.Resampling.BILINEAR
    )

    depth = np.asarray(
        imagem_depth,
        dtype=np.float32
    ) / 255.0

    return depth


def preparar_mascara(imagem):
    """
    Obtém a máscara do objeto.

    Se a imagem tiver canal alfa, usa o alfa.
    Caso contrário, considera toda a imagem como objeto.
    """

    if isinstance(imagem, Image.Image):

        if imagem.mode == "RGBA":

            mascara = np.asarray(
                imagem.getchannel("A"),
                dtype=np.uint8
            )

        else:

            mascara = np.ones(
                (imagem.height, imagem.width),
                dtype=np.uint8
            ) * 255

    else:

        imagem = np.asarray(imagem)

        if imagem.ndim == 3 and imagem.shape[2] == 4:

            mascara = imagem[:, :, 3]

        else:

            mascara = np.ones(
                imagem.shape[:2],
                dtype=np.uint8
            ) * 255

    return mascara


def depth_to_pointcloud(
    depth,
    image=None,
    mask=None,
    stride=2,
    invert_depth=True,
    focal_length=None
):
    """
    Converte mapa de profundidade em nuvem de pontos 3D.

    Retorna:

        points -> [N, 3]
        colors -> [N, 3]

    Parâmetros:

        depth:
            mapa de profundidade normalizado 0..1

        image:
            imagem RGB/RGBA usada para colorir os pontos

        mask:
            máscara do objeto

        stride:
            espaçamento entre pontos

        invert_depth:
            inverte o eixo Z

        focal_length:
            distância focal aproximada
    """

    depth = np.asarray(
        depth,
        dtype=np.float32
    )

    if depth.ndim != 2:

        depth = np.squeeze(
            depth
        )

    altura, largura = depth.shape

    # --------------------------------------------------------
    # IMAGEM
    # --------------------------------------------------------

    if image is not None:

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.fromarray(
                np.asarray(image)
            )

        image = image.convert(
            "RGB"
        )

        if image.size != (
            largura,
            altura
        ):

            image = image.resize(
                (largura, altura),
                Image.Resampling.BILINEAR
            )

        rgb = np.asarray(
            image,
            dtype=np.uint8
        )

    else:

        rgb = np.ones(
            (altura, largura, 3),
            dtype=np.uint8
        ) * 255

    # --------------------------------------------------------
    # MÁSCARA
    # --------------------------------------------------------

    if mask is None:

        mask_array = np.ones(
            (altura, largura),
            dtype=np.uint8
        ) * 255

    else:

        if isinstance(
            mask,
            Image.Image
        ):

            mask_image = mask.convert(
                "L"
            )

            mask_image = mask_image.resize(
                (largura, altura),
                Image.Resampling.NEAREST
            )

            mask_array = np.asarray(
                mask_image,
                dtype=np.uint8
            )

        else:

            mask_array = np.asarray(
                mask
            )

            if mask_array.ndim == 3:

                mask_array = mask_array[:, :, 0]

            mask_image = Image.fromarray(
                mask_array.astype(np.uint8),
                mode="L"
            )

            mask_image = mask_image.resize(
                (largura, altura),
                Image.Resampling.NEAREST
            )

            mask_array = np.asarray(
                mask_image,
                dtype=np.uint8
            )

    # --------------------------------------------------------
    # REDIMENSIONA DEPTH SE NECESSÁRIO
    # --------------------------------------------------------

    if depth.shape != mask_array.shape:

        depth = preparar_depth(
            depth,
            (largura, altura)
        )

    # --------------------------------------------------------
    # AMOSTRAGEM
    # --------------------------------------------------------

    ys = np.arange(
        0,
        altura,
        stride
    )

    xs = np.arange(
        0,
        largura,
        stride
    )

    grid_x, grid_y = np.meshgrid(
        xs,
        ys
    )

    grid_depth = depth[
        grid_y,
        grid_x
    ]

    grid_mask = mask_array[
        grid_y,
        grid_x
    ]

    grid_rgb = rgb[
        grid_y,
        grid_x
    ]

    # --------------------------------------------------------
    # SOMENTE OBJETO
    # --------------------------------------------------------

    valido = (
        grid_mask > 10
    )

    x_pixel = grid_x[
        valido
    ].astype(
        np.float32
    )

    y_pixel = grid_y[
        valido
    ].astype(
        np.float32
    )

    z_depth = grid_depth[
        valido
    ].astype(
        np.float32
    )

    colors = grid_rgb[
        valido
    ]

    # --------------------------------------------------------
    # PROFUNDIDADE
    # --------------------------------------------------------

    if invert_depth:

        z = 1.0 - z_depth

    else:

        z = z_depth.copy()

    # Evita pontos exatamente no zero
    z = z + 0.001

    # --------------------------------------------------------
    # CÂMERA
    # --------------------------------------------------------

    cx = (
        largura - 1
    ) / 2.0

    cy = (
        altura - 1
    ) / 2.0

    if focal_length is None:

        focal_length = float(
            max(
                largura,
                altura
            )
        )

    # --------------------------------------------------------
    # PROJEÇÃO PERSPECTIVA
    # --------------------------------------------------------

    x = (
        (x_pixel - cx)
        * z
        / focal_length
    )

    y = (
        (y_pixel - cy)
        * z
        / focal_length
    )

    # --------------------------------------------------------
    # CENTRALIZA
    # --------------------------------------------------------

    x -= x.mean()
    y -= y.mean()
    z -= z.mean()

    points = np.stack(
        [
            x,
            -y,
            z
        ],
        axis=1
    ).astype(
        np.float32
    )

    colors = colors.astype(
        np.uint8
    )

    return points, colors


def save_pointcloud_ply(
    path,
    points,
    colors
):
    """
    Salva a nuvem de pontos no formato PLY.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    colors = np.asarray(
        colors,
        dtype=np.uint8
    )

    if len(points) != len(colors):

        raise ValueError(
            "Quantidade de pontos e cores diferente."
        )

    with open(
        path,
        "w",
        encoding="ascii"
    ) as arquivo:

        arquivo.write(
            "ply\n"
        )

        arquivo.write(
            "format ascii 1.0\n"
        )

        arquivo.write(
            f"element vertex {len(points)}\n"
        )

        arquivo.write(
            "property float x\n"
        )

        arquivo.write(
            "property float y\n"
        )

        arquivo.write(
            "property float z\n"
        )

        arquivo.write(
            "property uchar red\n"
        )

        arquivo.write(
            "property uchar green\n"
        )

        arquivo.write(
            "property uchar blue\n"
        )

        arquivo.write(
            "end_header\n"
        )

        for ponto, cor in zip(
            points,
            colors
        ):

            arquivo.write(
                f"{ponto[0]:.6f} "
                f"{ponto[1]:.6f} "
                f"{ponto[2]:.6f} "
                f"{int(cor[0])} "
                f"{int(cor[1])} "
                f"{int(cor[2])}\n"
            )
