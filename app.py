import gradio as gr
from router import process_input

def handle_upload(file):
    if file is None:
        return 'Please upload an image or audio file.'
    return process_input(file)

demo = gr.Interface(
    fn=handle_upload,
    inputs=gr.File(
        label='Upload a patient image or a health worker voice note',
        file_types=['image', 'audio'],
        type='filepath',  # return a path string; older Gradio returns a
                          # temp-file wrapper without this, breaking splitext
    ),
    outputs=gr.Textbox(label='AfyaPlus Multimodal Assistant Output', lines=10),
    title='AfyaPlus Multimodal Intake Assistant',
    description=('Upload an image for a safe, non-diagnostic description, or '
                 'an audio file for a transcript. All image descriptions '
                 'include a mandatory clinical disclaimer.'),
)

if __name__ == '__main__':
    demo.launch()