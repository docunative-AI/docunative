#Author : Paarth Sharma
#Project : DocuNative 
#Filename : app.py  
#Start Date : 03-12-2026
#Modification Date : 03-12-2026
#Description : a gradio ui for the docunative project, allowing the user to upload a pdf, ask a question, select a model and get an answer 

#imports
import gradio as gr

#Pre : pdf_file, question, model_choice
#Post : answer, source clause, nli status
#Desc : function for handing the ask button click, currently returns demo values
def ask(pdf_file, question, model_choice):
    return (
        "[demo answer]",
        "[demo clause]",
        "Entailment",
    )

#Example questions for the UI 
EXAMPLES = [
    ["What is my deposit amount?"],
    ["When does my lease end?"],
    ["What is the notice period?"],
]

#Gradio block UI setup
with gr.Blocks(title="DocuNative") as demo:
    gr.Markdown("## DocuNative: Multilingual Document Q&A")

    with gr.Row():
        # Left column
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            question_input = gr.Textbox(label="Question", placeholder="Ask a question about your document")
            model_radio = gr.Radio(
                choices=["Global", "Earth"],
                value="Global",
                label="Select Model",
            )
            ask_btn = gr.Button("Ask", variant="primary")
            gr.Examples(
                examples=EXAMPLES,
                inputs=question_input,
                label="Example Questions",
            )

        # Right column
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=4)
            source_output = gr.Textbox(label="Source Clause", interactive=False, lines=4)
            nli_output = gr.Textbox(label="NLI Status", interactive=False, lines=2)

    ask_btn.click(
        fn=ask,
        inputs=[pdf_input, question_input, model_radio],
        outputs=[answer_output, source_output, nli_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
