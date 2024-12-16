from img_emb import *
from text_emb import *

def similarity(e1,e2):
    sim = torch.nn.CosineSimilarity(dim = 0)
    score = sim(e1,e2)
    return score

print(similarity(embeddings_i,embeddings_t))
