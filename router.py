import os
from caption_image import caption_image
from transcribe_audio import transcribe_audio

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg'}

def process_input(file_path):
    ext = os.path.splitext(file_path)[1].lower()  # split off the file extension
    if ext in IMAGE_EXTENSIONS:
        result = caption_image(file_path)
        return f'[Image Caption]\n\n{result}'
    elif ext in AUDIO_EXTENSIONS:
        transcript = transcribe_audio(file_path)
        return (f'[Audio Transcript - detected language: '
                f'{transcript.language}]\n\n{transcript.text}')
    elif ext == ".pdf":
        return ("PDF files are not supported yet. Please upload an image "
                "(.jpg/.png) or an audio file (.mp3/.wav).")

    else:
        return (f'Unsupported file type: {ext}. Please upload an image '
                f'(.jpg/.png) or audio file (.mp3/.wav).')