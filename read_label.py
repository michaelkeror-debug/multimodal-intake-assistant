from dotenv import load_dotenv
from openai import OpenAI
import os
import base64
import json
load_dotenv()  # read the API key from your .env file
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

DISCLAIMER = (
    'This description is for informational purposes only and is not a '
    'medical diagnosis. Please consult an AfyaPlus clinician for assessment.'
)
SYSTEM_PROMPT = '''You extract printed text from photos of medication packaging.
Return ONLY a JSON object with these fields:
  medication_name, strength, expiry_date, other_visible_text
If a field is not clearly readable, set it to "UNREADABLE" - never guess.
You do not give dosage advice or identify what a medication treats.'''

def encode_image(path):
    with open(path, 'rb') as f:
        image_bytes = f.read()                       # raw bytes of the image file
        b64_bytes = base64.b64encode(image_bytes)    # encode the bytes as base64
        return b64_bytes.decode('utf-8')             # images travel to the API as base64 text

def read_label(image_path):
    b64 = encode_image(image_path)   # encode the photo once, up front
    response = client.chat.completions.create(
        model='gpt-4o', temperature=0,  # deterministic: same input, same output
        response_format={'type': 'json_object'},  # force valid JSON output
        messages=[
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Extract the printed label fields.'},
                {'type': 'image_url', 'image_url': {
                    'url': f'data:image/jpeg;base64,{b64}'}},
            ]},
        ],
    )
    result =  json.loads(response.choices[0].message.content)  # parse the JSON string into a Python dict
    result['disclaimer'] = DISCLAIMER
    return result

if __name__ == '__main__':
    # run this file directly to read one label
    print(read_label('images/label_sharp1.jpeg'))
    print(read_label('images/label_blurry.jpeg'))
    print(read_label('images/label_severe.jpeg'))