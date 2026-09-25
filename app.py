import gradio as gr

from segmentation import remover_fundo
from depth.depth_engine import DepthEngine
from depth.depth_visualizer import depth_to_image


# ============================================================
# MOTOR DE PROFUNDIDADE
# ============================================================

depth_engine = DepthEngine()


# ============================================================
# PROCESSAMENTO
# ============================================================

def processar(imagem):

    if imagem is None:

        return (
            None,
            None,
            "Nenhuma imagem enviada."
        )

    try:

        # ====================================================
        # ETAPA 1 — SEGMENTAÇÃO
        # ====================================================

        objeto = remover_fundo(
            imagem
        )

        if objeto is None:

            return (
                None,
                None,
                "Erro na segmentação."
            )

        # ====================================================
        # ETAPA 2 — PROFUNDIDADE
        # ====================================================

        resultado_depth = depth_engine.analisar(
            objeto
        )

        mapa_depth = depth_to_image(
            resultado_depth.depth
        )

        # ====================================================
        # RESULTADO
        # ====================================================

        return (
            objeto,
            mapa_depth,
            "Processamento concluído."
        )

    except Exception as erro:

        return (
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

        **Imagem → Segmentação → Profundidade**

        O motor de profundidade será desenvolvido
        pela própria Solaria3D.
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


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    app.launch()
