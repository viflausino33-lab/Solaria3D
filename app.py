import gradio as gr

from segmentation import remover_fundo
from depth.test_depth_engine import TestDepthEngine
from depth.depth_visualizer import depth_to_image


# Motor de profundidade de teste
depth_engine = TestDepthEngine()


def processar(imagem):
    if imagem is None:
        return None, None, "Nenhuma imagem enviada."

    try:
        # ==========================================
        # ETAPA 1 — SEGMENTAÇÃO
        # ==========================================
        objeto = remover_fundo(imagem)

        if objeto is None:
            return None, None, "Erro na segmentação."

        # ==========================================
        # ETAPA 2 — PROFUNDIDADE
        # ==========================================
        resultado_depth = depth_engine.analisar(objeto)

        mapa_depth = depth_to_image(resultado_depth.depth)

        return (
            objeto,
            mapa_depth,
            "Processamento concluído: segmentação + profundidade."
        )

    except Exception as erro:
        return None, None, f"Erro: {erro}"


with gr.Blocks(title="Solaria3D") as app:

    gr.Markdown(
        """
        # Solaria3D

        **Transformação de imagens 2D em modelos 3D**

        ### Pipeline atual

        **1. Segmentação → 2. Profundidade**

        A profundidade ainda utiliza um motor de teste.
        Posteriormente será substituído pelo modelo real.
        """
    )

    imagem = gr.Image(
        type="pil",
        label="Imagem 2D"
    )

    botao = gr.Button(
        "Processar imagem",
        variant="primary"
    )

    with gr.Row():

        objeto = gr.Image(
            type="pil",
            label="1 — Objeto segmentado"
        )

        mapa_depth = gr.Image(
            type="pil",
            label="2 — Mapa de profundidade"
        )

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    botao.click(
        fn=processar,
        inputs=imagem,
        outputs=[
            objeto,
            mapa_depth,
            status
        ]
    )


app.launch()
