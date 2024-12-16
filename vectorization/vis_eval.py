from img_emb import *
from text_emb import *
import matplotlib.pyplot as plt

def similarity(e1,e2):
    sim = torch.nn.CosineSimilarity(dim = 0)
    score = sim(e1,e2)
    return score

print(similarity(embeddings_i,embeddings_t))

def vis(image_embedding, text_embedding):

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(image_embedding)), image_embedding.detach().numpy())
    plt.title('Image Embedding')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(text_embedding)), text_embedding.detach().numpy())
    plt.title('Text Embedding')
    plt.xlabel('Dimension')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()

vis(embeddings_i,embeddings_t)