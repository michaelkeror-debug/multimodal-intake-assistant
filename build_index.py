import os
import torch
import open_clip
import pickle
from PIL import Image
if torch.cuda.is_available():
    device = 'cuda'   # use the GPU if one is available
else:
    device = 'cpu'    # otherwise fall back to the CPU
# load the pretrained CLIP model (weights download on first run)
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='openai')
model = model.to(device).eval()
IMAGE_DIR = 'images'
if not os.path.isdir(IMAGE_DIR):
    print(f'Folder not found: {IMAGE_DIR}/')
    raise SystemExit(1)
image_paths = []
for f in os.listdir(IMAGE_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_paths.append(os.path.join(IMAGE_DIR, f))
image_vectors = []
valid_paths = []
with torch.no_grad():
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')   # open the file as an RGB image
            img = preprocess(image)                   # resize and normalise for CLIP
            img = img.unsqueeze(0).to(device)         # add a batch dimension, move to device
            vec = model.encode_image(img)  # image becomes a 512-number vector
            vec = vec / vec.norm(dim=-1, keepdim=True)  # normalise for cosine
            image_vectors.append(vec.cpu())
            valid_paths.append(path)
        except Exception as e:
            print(f'Skipping {path}: {e}')
if not image_vectors:
    print('No images indexed.')
    raise SystemExit(1)
image_vectors = torch.cat(image_vectors, dim=0)
with open('image_index.pkl', 'wb') as f:
    # save the index so searches never re-encode the library
    pickle.dump({'paths': valid_paths, 'vectors': image_vectors}, f)
print(f'Indexed {len(valid_paths)} images into image_index.pkl')