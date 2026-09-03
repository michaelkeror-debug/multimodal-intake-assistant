import torch
import open_clip
import pickle
if torch.cuda.is_available():
    device = 'cuda'   # use the GPU if one is available
else:
    device = 'cpu'    # otherwise fall back to the CPU
# load the pretrained CLIP model (weights download on first run)
model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')  # prepares text for the text encoder
model = model.to(device).eval()
with open('image_index.pkl', 'rb') as f:
    index = pickle.load(f)  # load the saved index from disk

def search_best_match(query_text):
    with torch.no_grad():
        tokens = tokenizer([query_text]).to(device)
        query_vec = model.encode_text(tokens)  # text becomes a 512-number vector
        query_vec = query_vec / query_vec.norm(dim=-1, keepdim=True)
        
    similarities = index['vectors'] @ query_vec.T.cpu()   # one score per image
    similarities = similarities.squeeze(1)                # flatten to a plain list of scores
    best_index = similarities.argmax().item()
    best_score = similarities[best_index].item()

    best_path = index['paths'][best_index]

    # Convert similarity to percentage
    percentage = best_score * 100

    return best_path, percentage

if __name__ == '__main__':
    query = 'a poster about physical exercise'

    path, score = search_best_match(query)

    print(f'Query: {query}')
    print(f'Best match: {path}')
   
