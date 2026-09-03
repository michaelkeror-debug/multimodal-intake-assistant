from dotenv import load_dotenv
from openai import OpenAI
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EXTRACT_PROMPT = '''From the transcript, return ONLY JSON with fields:
symptoms (list),
follow_up_priority (routine | supervisor_review | same_day_callback),
requires_clinical_review (true/false).

Base every field on the transcript only.
Do not infer a diagnosis.
'''

TRIGGERS = [
    'children under five',
    'difficulty breathing',
    'vomitting'
]


def transcribe_audio(audio_path):
    with open(audio_path, 'rb') as audio_file:
        transcript = client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            response_format='verbose_json'
        )

    return transcript


def extract_fields(transcript_text):
    r = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0,
        response_format={'type': 'json_object'},
        messages=[
            {
                'role': 'system',
                'content': EXTRACT_PROMPT
            },
            {
                'role': 'user',
                'content': transcript_text
            }
        ]
    )

    return json.loads(r.choices[0].message.content)


if __name__ == '__main__':

    # 1. Transcribe audio
    transcript = transcribe_audio('sample_report.mp3')

    # 2. Get the actual transcript text
    transcript_text = transcript.text

    print('Detected language:', transcript.language)
    print('Transcript:', transcript_text)

    # 3. Extract structured fields
    result = extract_fields(transcript_text)

    # 4. Display extracted information
    print('\nExtracted fields:')
    print(json.dumps(result, indent=2))