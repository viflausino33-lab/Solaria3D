import gradio as gr


def analisar(imagem):
    if imagem is None:
        return "Nenhuma imagem enviada."

    return "Imagem recebida com sucesso."


with gr.Blocks(title="Solaria3D") as app:

    gr.Markdown(
        """
        # Solaria3D

        **Transformação de imagens 2D em modelos 3D**

        Primeira versão do sistema.
        """
    )

    imagem = gr.Image(
        type="pil",
        label="Imagem 2D"
    )

    resultado = gr.Textbox(
        label="Status"
    )

    botao = gr.Button(
        "Analisar imagem"
    )

    botao.click(
        fn=analisar,
        inputs=imagem,
        outputs=resultado
    )


app.launch()
