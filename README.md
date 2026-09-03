# multimodal-intake-assistant

# AfyaPlus Multimodal Intake Assistant

## Overview
Readiness assessment of CLIP, Whisper and unified Gradio application for AfyaPlus Healthcare

## Project Structure
- images/: Sample images used for the retrieval index and captioning tests.
- build_index.py: Builds the CLIP retrieval index.
- search.py: Text-to-image search function.
- caption_image.py: GPT-4o vision captioning pipeline.
- transcribe_audio.py: Whisper transcription pipeline.
- router.py: Input-routing logic.
- app.py: Gradio application entry point.

## How to Reproduce
1. Environment setup: pip install -r requirements.txt
2. Build the index: python build_index.py
3. Launch the app: python app.py and open the printed local URL

## Safety Note
- Image captions are for informational and operational purposes only
  and are not a medical diagnosis.