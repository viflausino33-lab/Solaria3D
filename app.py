import gradio as gr

from segmentation import remover_fundo


def analisar(imagem):
    if imagem is None:
        return None, "Nenhuma imagem enviada."

    try:
        resultado = remover_fundo(imagem)

        return resultado, "Segmentação concluída."

    except Exception as erro:
        return None, f"Erro na segmentação: {erro}"


with gr.Blocks(title="Solaria3D") as app:

    gr.Markdown(
        """
        # Solaria3D

        **Transformação de imagens 2D em modelos 3D**

        ### Etapa 1 — Segmentação
        O sistema identifica o objeto e remove o fundo da imagem.
        """
    )

    with gr.Row():

        imagem = gr.Image(
            type="pil",
            label="Imagem 2D"
        )

        resultado = gr.Image(
            type="pil",
            label="Objeto sem fundo"
        )

    status = gr.Textbox(
        label="Status",
        interactive=False
    )

    botao = gr.Button(
        "1. Segmentar imagem",
        variant="primary"
    )

    botao.click(
        fn=analisar,
        inputs=imagem,
        outputs=[resultado, status]
    )


app.launch()
