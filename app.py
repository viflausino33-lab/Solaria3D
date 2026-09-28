import os

import gradio as gr

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine
from depth.depth_visualizer import depth_to_image
from depth.pointcloud import (
    depth_to_pointcloud,
    save_pointcloud_ply
)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

MODEL_PATH = "models/solaria_depth_v1.ply"


# ============================================================
# MOTOR DE PROFUNDIDADE
# ============================================================

depth_engine = SolariaDepthEngine()


# ============================================================
# PROCESSAMENTO
# ============================================================

def processar(imagem):

    if imagem is None:

        return (
            None,
            None,
            None,
            "Nenhuma imagem enviada."
        )

    try:

        # ====================================================
        # ETAPA 1 — SEGMENTAÇÃO
        # ====================================================

        print()
        print("Iniciando segmentação...")

        objeto = remover_fundo(
            imagem
        )

        if objeto is None:

            return (
                None,
                None,
                None,
                "Erro na segmentação."
            )

        print(
            "Segmentação concluída."
        )


        # ====================================================
        # ETAPA 2 — PROFUNDIDADE
        # ====================================================

        print(
            "Gerando profundidade..."
        )

        resultado_depth = depth_engine.analisar(
            objeto
        )

        print(
            "Profundidade concluída."
        )


        # ====================================================
        # ETAPA 3 — VISUALIZAÇÃO DO DEPTH
        # ====================================================

        mapa_depth = depth_to_image(
            resultado_depth
        )


        # ====================================================
        # ETAPA 4 — NUVEM DE PONTOS
        # ====================================================

        print(
            "Gerando nuvem de pontos..."
        )

        points, colors = depth_to_pointcloud(
            resultado_depth,
            image=objeto,
            mask=objeto,
            stride=2
        )

        print(
            "Pontos gerados:",
            len(points)
        )


        # ====================================================
        # ETAPA 5 — SALVAR PLY
        # ====================================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        save_pointcloud_ply(
            MODEL_PATH,
            points,
            colors
        )

        print(
            "Nuvem salva:",
            MODEL_PATH
        )


        # ====================================================
        # RESULTADO
        # ====================================================

        return (
            objeto,
            mapa_depth,
            MODEL_PATH,
            (
                "Processamento concluído. "
                f"{len(points):,} pontos 3D gerados."
            )
        )


    except Exception as erro:

        print()
        print(
            "ERRO:",
            type(erro).__name__,
            erro
        )

        return (
            None,
            None,
            None,
            f"Erro: {type(erro).__name__}: {erro}"
        )


# ============================================================
# INTERFACE
# ============================================================

with gr.Blocks(
    title="Solaria3D"
) as app:

    gr.Markdown(
        """
        # Solaria3D

        **Transformação de imagens 2D em modelos 3D**

        ### Pipeline atual

        **Imagem → Segmentação → Profundidade → Nuvem de pontos 3D**

        A profundidade é estimada pela rede neural
        própria da Solaria3D.
        """
    )


    # ========================================================
    # ENTRADA
    # ========================================================

    imagem = gr.Image(
        type="pil",
        label="Imagem 2D"
    )


    botao = gr.Button(
        "Processar imagem",
        variant="primary"
    )


    # ========================================================
    # RESULTADOS 2D
    # ========================================================

    with gr.Row():

        objeto = gr.Image(
            type="pil",
            label="1 — Objeto segmentado"
        )

        mapa_depth = gr.Image(
            type="pil",
            label="2 — Mapa de profundidade"
        )


    # ========================================================
    # RESULTADO 3D
    # ========================================================

    modelo_3d = gr.Model3D(
        label="3 — Nuvem de pontos 3D",
        display_mode="point_cloud",
        height=600,
        camera_position=(
            45,
            25,
            3
        )
    )


    # ========================================================
    # STATUS
    # ========================================================

    status = gr.Textbox(
        label="Status",
        interactive=False
    )


    # ========================================================
    # EVENTO
    # ========================================================

    botao.click(
        fn=processar,
        inputs=imagem,
        outputs=[
            objeto,
            mapa_depth,
            modelo_3d,
            status
        ]
    )


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.launch()
