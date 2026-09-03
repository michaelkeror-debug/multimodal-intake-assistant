import torch
import open_clip
import pickle
from  PIL import Image
if torch.cuda.is_available():
    device = 'cuda'   # use the GPU if one is available
else:
    device = 'cpu'    # otherwise fall back to the CPU
# load the pretrained CLIP model (weights download on first run)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')  # prepares text for the text encoder
model = model.to(device).eval()
with open('image_index.pkl', 'rb') as f:
    index = pickle.load(f)  # load the saved index from disk

def search_by_image(query_path, top_k=3):
    
    with torch.no_grad():
        image = preprocess(Image.open(query_path)).unsqueeze(0).to(device)
        query_vec = model.encode_image(image)  # text becomes a 512-number vector
        query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
    similarities = index['vectors'] @ query_vec.T.cpu()   # one score per image
    similarities = similarities.squeeze(1)                # flatten to a plain list of scores
    top = similarities.topk(min(top_k, len(index['paths'])))  # take the k best matches
    results = []
    for sc, i in zip(top.values, top.indices):
        results.append((index['paths'][i], float(sc)))
    return results

if __name__ == '__main__':
   query_path = "images/poster_12.jpeg"
   result = search_by_image(query_path, top_k=10)
   ranked = sorted(result, key= lambda x:x[1], reverse=True)

   #print("=== raw ranked results ===")
   #for path, score in ranked:
       #print(f"{score:.4f} {path}")
# query_path against every image in the saved index
   matches = []
for path, score in ranked:
    if path != query_path and score > 0.90:               # keep only near-duplicates
        matches.append((path, score))
print(matches)                  