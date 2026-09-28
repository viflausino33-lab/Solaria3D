import gradio as gr

from segmentation import remover_fundo
from depth.solaria_engine import SolariaDepthEngine
from depth.depth_visualizer import depth_to_image


# ============================================================
# MOTOR DE PROFUNDIDADE — SOLARIADEPTH
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
                "Erro na segmentação."
            )

        print("Segmentação concluída.")

        # ====================================================
        # ETAPA 2 — SOLARIADEPTH
        # ====================================================

        print("Executando SolariaDepth...")

        depth = depth_engine.analisar(
            objeto
        )

        print(
            "Depth gerado:",
            depth.shape
        )

        print(
            "Depth min:",
            depth.min()
        )

        print(
            "Depth max:",
            depth.max()
        )

        # ====================================================
        # ETAPA 3 — VISUALIZAÇÃO
        # ====================================================

        mapa_depth = depth_to_image(
            depth
        )

        print("Mapa de profundidade gerado.")

        # ====================================================
        # RESULTADO
        # ====================================================

        return (
            objeto,
            mapa_depth,
            "Processamento concluído com SolariaDepth."
        )

    except Exception as erro:

        print()
        print("ERRO:")
        print(
            type(erro).__name__,
            erro
        )

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

        **Imagem → Segmentação → SolariaDepth**

        A profundidade é estimada pela rede neural
        própria da Solaria3D.
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
