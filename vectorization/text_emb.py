from transformers import AutoTokenizer, AutoModel
from img_emb import *

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
encoder = AutoModel.from_pretrained('bert-base-uncased')

text = "this is an man playing with fire"
tokens = tokenizer(
    text,
    return_tensors = 'pt',
    padding = True,
    truncation = True,
    max_length = 512
)

with torch.no_grad():
    encodings = encoder(**tokens)
    text_embeddings = encodings.last_hidden_state[:,0,:] # i extract the [cls] token
    
conv = torch.nn.Linear(768,300)
embeddings_t = conv(text_embeddings).squeeze()
print(embeddings_t.shape)