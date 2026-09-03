from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
load_dotenv()  # read the API key from your .env file
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Single source of truth - imported by the disclaimer check in Challenge 1
DISCLAIMER = (
    'This description is for informational purposes only and is not a '
    'medical diagnosis. Please consult an AfyaPlus clinician for assessment.'
)
SYSTEM_PROMPT = f'''You are the AfyaPlus image captioning assistant. \
You describe images submitted by patients or community health workers \
in clear, factual, non-diagnostic language.
Rules you must always follow:
- Describe only what is visibly present (colour, location, size, texture).
- Never state or imply a medical diagnosis, condition name, or severity.
- Never recommend a specific medication or treatment.
- Always end your response with exactly this sentence: '{DISCLAIMER}'
- If the image is unclear or shows no visible health concern, say so plainly.'''


def encode_image(path):
    # Step 1: read the image and turn its bytes into a base64 string
    with open(path, 'rb') as f:
        image_bytes = f.read()                       # raw bytes of the image file
        b64_bytes = base64.b64encode(image_bytes)    # encode the bytes as base64
        return b64_bytes.decode('utf-8')             # bytes become plain text

def caption_image(image_path):
    b64 = encode_image(image_path)
    # Step 2: send the system rules, the instruction, and the image together
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},   # the safety contract
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Describe this image for our clinical operations log.'},
                {'type': 'image_url', 'image_url': {         # the image rides inside the message
                    'url': f'data:image/jpeg;base64,{b64}'}},
            ]},
        ],
        max_tokens=300,
        temperature=0.2,
    )
    return response.choices[0].message.content

if __name__ == '__main__':
    print(caption_image('images/poster_11.jpeg'))
