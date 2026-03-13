#Author : Paarth Sharma 
# Updated: Vinod Anbalagan — port config, null check, NLI badge component
#Project : DocuNative 
#Filename : app.py  
#Start Date : 03-12-2026
#Modification Date : 03-12-2026
#Description : a gradio ui for the docunative project, allowing the user to upload a pdf, ask a question, select a model and get an answer 

#imports
import os
import gradio as gr

# Read port from environment variable — default 7860
# Override by running: GRADIO_PORT=7861 make demo
PORT = int(os.getenv("GRADIO_PORT", 7860))

# Pipeline imports (wired in Issue #24) 
# from pipeline.extract import extract_and_chunk # for issue #4
# from pipeline.embed   import embed_and_store   # for issue #14
# from pipeline.retrieve import retrieve         # for issue #14
# from pipeline.generate import generate_answer  # for issue #15
# from pipeline.validate import parse_output     # for issue #7
# from pipeline.nli      import classify_nli    # for issue #19

#Example questions for the UI 
EXAMPLES = [
    ["What is my deposit amount?"],
    ["When does my lease end?"],
    ["What is the notice period?"],
    ["What are the rules about pets?"],
    ["Who is responsible for repairs?"],
]

# NLI badge HTML helper 
# Returns a colour-coded HTML badge for the NLI result.
# Issue #25 will call classify_nli() and pass the label here.
def nli_badge(label: str) -> str:
    colours = {
        "Entailment":    ("#d5f5e3", "#1e8449", "✅ Entailment — answer is supported"),
        "Neutral":       ("#fef9e7", "#b7950b", "⚠️ Neutral — answer may be incomplete"),
        "Contradiction": ("#fadbd8", "#922b21", "🚨 Contradiction — possible hallucination"),
        "N/A":           ("#f0f0f0", "#888888", "— Upload a PDF and ask a question"),
    }
    bg, fg, text = colours.get(label, colours["N/A"])
    return (
        f'<div style="background:{bg}; color:{fg}; padding:12px 16px; '
        f'border-radius:8px; font-weight:600; font-size:14px;">{text}</div>'
    )


# --- Core handler ---
#Pre : pdf_file, question, model_choice
#Post : answer, source clause, nli status
#Desc : function for handing the ask button click, currently returns demo values
def ask(pdf_file, question, model_choice, ui_language):
    # Guard: PDF not yet uploaded
    if pdf_file is None:
        return (
            "Please upload a PDF first.",
            "",
            nli_badge("N/A"),
        )

    # Guard: question is empty
    if not question or not question.strip():
        return (
            "Please type a question.",
            "",
            nli_badge("N/A"),
        )

    # Demo stubs — replace in Issue #24 
    demo_answer = "[demo answer — pipeline not yet wired]"
    demo_quote  = "[demo source quote]"
    demo_nli    = "Entailment"  # will come from classify_nli() in Issue #25
    # ----------

    return (
        demo_answer,
        demo_quote,
        nli_badge(demo_nli),
    )


#Gradio block UI setup
with gr.Blocks(title="DocuNative", theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.Markdown(
            "## DocuNative — Multilingual Document Q&A\n"
            "Upload a foreign-language document. Ask a question in your own language. "
            "Get an answer — **entirely on your device.**"
        )
        # ADDED: Olena's UI Language Selector (makes it easier for pahse 2) 
        ui_lang_dropdown = gr.Dropdown(
            choices=["English", "Swahili", "Hindi", "German", "Arabic", "French"],
            value="English",
            label="App Language",
            interactive=True,
            scale=0,
            min_width=150
        )

    with gr.Row():
        # Left column: inputs
        with gr.Column(scale =1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf",".txt"])
            question_input = gr.Textbox(
                label="Question", 
                placeholder="Ask a question about your document", 
                lines = 2,
                )
            model_radio = gr.Radio(
                choices=["Global", "Earth"],
                value="Global",
                label="Select Model",
                info="Global = multilingual generalist | Earth = Africa & West Asia specialist",
            )
            ask_btn = gr.Button("Ask DocuNative", variant="primary", size="lg")
            gr.Examples(
                examples=EXAMPLES,
                inputs=question_input,
                label="Example Questions",
            )

        # Right column : Outputs 
        with gr.Column():
            answer_output = gr.Textbox(label="Answer", interactive=False, lines=5, placeholder="Your answer will appear here..." )
            source_output = gr.Textbox(label="Source Clause", interactive=False, lines=4, placeholder="The supporting quote from the document will appear here...")
            with gr.Group():
                gr.Markdown("**Trust Score (Hallucination Check)**")
                nli_output = gr.HTML(value=nli_badge("N/A"))

    ask_btn.click(
        fn=ask,
        inputs=[pdf_input, question_input, model_radio, ui_lang_dropdown],
        outputs=[answer_output, source_output, nli_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=PORT)
