"""
ui/app.py
---------
Gradio UI for DocuNative.

This file has ONE job: connect the user interface to the pipeline.
All AI logic lives in pipeline/pipeline.py — this file just handles
buttons, inputs, outputs, and display formatting.

Standalone run:
    python -m ui.app
    → opens at http://localhost:7860
    (requires llama-server running in Terminal 1: make server-global)
"""

#Author : Paarth Sharma 
# Updated: Vinod Anbalagan — port config, null check, NLI badge component
#Project : DocuNative 
#Filename : app.py  
#Start Date : 03-12-2026
#Modification Date : 03-16-2026

#imports
import os
import gradio as gr

# Privacy: Disable Gradio analytics — privacy-first app, no telemetry
os.environ["GRADIO_ANALYTICS_ENABLED"] = "false"

# Read port from environment variable — default 7860
# Override by running: GRADIO_PORT=7861 make demo
PORT = int(os.getenv("GRADIO_PORT", 7860))

# Pipeline import — single entry point for all 6 modules
# extract → embed → retrieve → generate → validate → nli
from pipeline.pipeline import run, PipelineResult

#Example questions for the UI 
EXAMPLES = [
    ["What is my deposit amount?"],
    ["When does my lease end?"],
    ["What is the notice period?"],
    ["What are the rules about pets?"],
    ["Who is responsible for repairs?"],
]

# Custom theme
# Uses system fonts only — no Google Fonts CDN calls (privacy-first).
docunative_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    font=["system-ui", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="*neutral_200",
    block_radius="xl",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)


# Custom CSS for the hero header and spacing
CUSTOM_CSS = """
.hero-header { text-align: center; margin-bottom: 20px; padding: 20px 0; }
.hero-title { font-weight: 800; font-size: 2.5em; color: #1e293b; margin-bottom: 8px; }
.hero-subtitle { font-size: 1.1em; color: #64748b; font-weight: 400; }
"""

# NLI badge HTML helper 
# Returns a colour-coded HTML badge for the NLI result.
# Issue #25 will call classify_nli() and pass the label here.
def nli_badge(label: str) -> str:
    colours = {
        "Entailment":    ("#ecfdf5", "#059669", "✅ Grounded — Answer is supported by the document"),
        "Neutral":       ("#fef2f2", "#d97706", "⚠️ Unverified — Could not confirm from source"),
        "Contradiction": ("#fef2f2", "#dc2626", "🚨 Hallucination — Answer conflicts with document"),
        "N/A":           ("#f8fafc", "#94a3b8", "— Upload a PDF and ask a question"),
    }
    bg, fg, text = colours.get(label, colours["N/A"])
    return f'<div style="background:{bg}; color:{fg}; padding:14px 16px; border-radius:10px; font-weight:600; font-size:15px; text-align:center; border: 1px solid {fg}33;">{text}</div>'


# Source quote highlighter
# Takes the source_quote and wraps it in a yellow highlight inside a
# document-styled div.

def highlight_quote(context: str, quote: str) -> str:
    """
    Wraps the source quote in a yellow highlight inside a document-styled div.

    Args:
        context: The text passage to display (currently same as quote)
        quote:   The exact quote to highlight within the context

    Returns:
        HTML string with the quote highlighted in yellow
    """
    if not context:
        return ""

    if quote and quote != "N/A" and quote in context:
        highlighted = context.replace(
            quote,
            f"<mark style='background-color: #fef08a; padding: 3px 6px; border-radius: 4px; "
            f"font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>{quote}</mark>"
        )
    else:
        highlighted = context

    # Styled to look like a physical document snippet (serif font, left border)
    return (
        f"<div style='white-space: pre-wrap; font-family: Georgia, serif; font-size: 15px; "
        f"line-height: 1.7; padding: 16px; background: #ffffff; border-radius: 0 8px 8px 0; "
        f"border-left: 4px solid #cbd5e1; color: #334155;'>{highlighted}</div>"
    )

# ---------------------
# Core ask() handler
# Called every time the user clicks "Ask DocuNative".
# Returns 4 values that map to the 4 output components.
def ask(pdf_file, question, model_choice, ui_language):
    """
    Handle the Ask button click.

    Args:
        pdf_file:     Gradio file object from the PDF upload component
        question:     User's question string (any language)
        model_choice: "Global" or "Earth" — which Tiny Aya variant to use
        ui_language:  Selected app interface language (not yet wired to translation)

    Returns:
        Tuple of (answer_text, nli_badge_html, context_html, parse_warning_html)
    """
    # Guard: PDF not yet uploaded
    if pdf_file is None:
        return "Please upload a PDF first.", nli_badge("N/A"), "", gr.update(visible=False)


    # Guard: question is empty
    if not question or not question.strip():
        return "Please type a question.", nli_badge("N/A"), "", gr.update(visible=False)

    # Run the full pipeline
    # force_reindex=False allows caching — the index is only rebuilt when
    # a new PDF is uploaded (Gradio gives each upload a unique temp path).
    # Setting True here would re-embed on EVERY click — destroying performance.
    result: PipelineResult = run(
        pdf_path=pdf_file.name,
        question=question,
        model_choice=model_choice,
        force_reindex=True,  # always reindex on each submit — prevents stale
                             # cache when user re-uploads a different PDF with
                             # the same filename (Gradio reuses the temp path).
    )

    # Pipeline error (e.g. llama-server not running)
    if result.error:
        return f"❌ {result.error}", nli_badge("N/A"), "", gr.update(visible=False)

    # Map pipeline NLI verdict to badge label
    nli_label_map = {
        "entailment":    "Entailment",
        "neutral":       "Neutral",
        "contradiction": "Contradiction",
    }
    # Parse warning — shown when model didn't follow the Answer:/Source_Quote: format
    # parse_success=False means we still got an answer but it may be less reliable
    warning_html = ""
    if not result.parse_success:
        warning_html = (
            "<div style='color: #92400e; background-color: #fef3c7; "
            "border-left: 4px solid #f59e0b; padding: 12px; border-radius: 4px; "
            "margin-bottom: 12px; font-size: 14px;'>"
            "⚠️ <b>Formatting Warning:</b> The AI's output was slightly malformed. "
            "We attempted to extract the answer, but please verify carefully.</div>"
        )

    # Phase 3 TODO: add retrieved_context field to PipelineResult for full chunk display.
    return (
        result.answer,
        nli_badge(nli_label_map.get(result.nli_verdict, "N/A")),
        # Pass the full retrieved context as the base text so the source
        # quote is highlighted within the actual document passage.
        # context_text is stable per query — fixes the shifting context bug.
        highlight_quote(result.context_text, result.source_quote),
        gr.update(value=warning_html, visible=not result.parse_success),
    )


#Gradio block UI setup
with gr.Blocks(title="DocuNative", theme=docunative_theme, css=CUSTOM_CSS) as demo:

    # Hero header
    gr.HTML("""
        <div class="hero-header">
            <div class="hero-title">🌍 DocuNative</div>
            <div class="hero-subtitle">Privacy-first, 100% on-device document intelligence for everyone.</div>
        </div>
    """)

    # Language selector (top right)
    # Note: not yet wired to actual translation — placeholder for Phase 3
    with gr.Row():
        gr.HTML("<div></div>")  # Spacer
        ui_lang_dropdown = gr.Dropdown(
            choices=["English", "Swahili", "Hindi", "German", "Arabic", "French"],
            value="English",
            label="App Interface Language",
            interactive=True,
            scale=0,
            min_width=200,
        )

    # Main two-column layout
    with gr.Row():

        # LEFT COLUMN — inputs
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📄 1. Upload & Ask")

            pdf_input = gr.File(
                label="Upload Legal Document",
                file_types=[".pdf", ".txt"],
            )
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., How much is the security deposit?",
                lines=2,
            )
            model_radio = gr.Radio(
                choices=["Global", "Earth"],
                value="Global",
                label="AI Model Engine",
                info="Earth = Africa/West Asia regional specialist",
            )

            ask_btn = gr.Button("Ask DocuNative", variant="primary", size="lg")

            with gr.Accordion("Need inspiration? Try these examples:", open=False):
                gr.Examples(examples=EXAMPLES, inputs=question_input, label="")

        # RIGHT COLUMN — outputs
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### ✨ 2. AI Analysis")

            # Parse warning — hidden by default, shown when parse_success=False
            parse_warning = gr.HTML(visible=False)

            # Answer box
            answer_output = gr.Textbox(
                label="Answer (in your language)",
                interactive=False,
                lines=3,
            )

            # NLI trust score badge
            gr.Markdown("<span style='font-size: 14px; font-weight: 600; color: #475569;'>Trust Score (Hallucination Check)</span>")
            nli_output = gr.HTML(value=nli_badge("N/A"))

            gr.Markdown("<br>")  # Spacer

            # Source context with highlighted quote
            gr.Markdown("<span style='font-size: 14px; font-weight: 600; color: #475569;'>Retrieved Document Context</span>")
            context_output = gr.HTML(
                value="<div style='color: #94a3b8; font-style: italic;'>The supporting quote will be highlighted here...</div>"
            )

    # Footer — privacy statement
    gr.HTML(
        "<div style='text-align: center; margin-top: 30px; color: #94a3b8; font-size: 12px;'>"
        "Built during the Cohere AI Hackathon • 100% Offline • Data never leaves your device"
        "</div>"
    )

    # Wire the Ask button to the handler
    ask_btn.click(
        fn=ask,
        inputs=[pdf_input, question_input, model_radio, ui_lang_dropdown],
        outputs=[answer_output, nli_output, context_output, parse_warning],
    )

# Entry point
# server_name="127.0.0.1" = localhost only — no external network access
# share=False (default) = no Gradio tunnel to HuggingFace servers    
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=PORT,
    )
