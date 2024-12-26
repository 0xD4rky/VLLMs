###### ---------  EMBEDDING SEARCH VIA FAISS INDEX --------- ######

import torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")

from datasets import load_dataset

dataset = load_dataset("jmhessel/newyorker_caption_contest", "explanation")

def vis_data(dataset):

    print(f"dataset dictionary : {dataset}")
    print(f"train dataset : {dataset['train']}")
    print(f"example image of train dataset : {dataset['train'][0]['image']}")
    print(f"example caption of train dataset : {dataset['train'][0]['image_description']}")

# if i need to visualize data --> vis_data(dataset)

