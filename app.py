import gradio as gr
from transformers import pipeline

# Load summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to summarize text or uploaded file
def summarize_text(text, file):
    # Read text from file if uploaded
    if file is not None:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()

    if text is None or len(text.split()) < 30:
        return "Please enter at least 30 words."

    # Summarize with truncation to prevent index errors
    summary = summarizer(
        text,
        max_length=80,
        min_length=30,
        do_sample=False,
        truncation=True
    )

    return summary[0]["summary_text"]

# Gradio Blocks UI
with gr.Blocks(css="""
    /* Input box styling - Light Red */
    #input-box textarea {
        background-color: #FFCCCC;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }

    /* Output box styling - Green */
    #output-box textarea {
        background-color: #00FF00;
        color: black;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }

    /* Button styling */
    button {
        font-size: 16px;
        font-weight: bold;
    }
""") as demo:

    gr.Markdown(
        """
        # ✨ GenAI Text Summarizer ✨
        Summarize long text or .txt files instantly using AI (BART model)
        """
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=12,
                placeholder="Paste text here...",
                label="Input Text",
                elem_id="input-box"
            )
            input_file = gr.File(
                file_types=[".txt"],
                label="Upload .txt file"
            )
            submit_btn = gr.Button("Generate Summary", variant="primary")
        
        with gr.Column():
            output_summary = gr.Textbox(
                lines=12,
                label="Summary Output",
                placeholder="Your summarized text will appear here...",
                elem_id="output-box"
            )

    submit_btn.click(
        summarize_text,
        inputs=[input_text, input_file],
        outputs=output_summary
    )

# Launch app with shareable link
demo.launch(share=True)
