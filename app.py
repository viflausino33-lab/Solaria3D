import gradio as gr

from segmentation import remover_fundo


def analisar(imagem):
    if imagem is None:
        return None, "Nenhuma imagem enviada."

    try:
        resultado = remover_fundo(imagem)

        return resultado, "Segmentação concluída com sucesso!"

    except Exception as erro:
        return None, f"Erro durante a segmentação: {erro}"


with gr.Blocks(title="Solaria3D") as app:

    gr.Markdown(
        """
        # Solaria3D

        **Transformação de imagens 2D em modelos 3D**

        ### Etapa 1 — Separação do objeto
        Envie uma imagem para remover o fundo.
        """
    )

    imagem = gr.Image(
        type="pil",
        label="Imagem 2D"
    )

    botao = gr.Button(
        "Analisar imagem"
    )

    resultado = gr.Image(
        type="pil",
        label="Objeto isolado"
    )

    status = gr.Textbox(
        label="Status"
    )

    botao.click(
        fn=analisar,
        inputs=imagem,
        outputs=[resultado, status]
    )


app.launch()
