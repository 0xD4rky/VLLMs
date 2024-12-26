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
    features = ['image', 'contest_number', 'image_location', 'image_description', 'image_uncanny_description', 'entities', 'questions', 'caption_choices', 'from_description', 'label', 'n_tokens_label', 'instance_id']
    for i,ff in enumerate(features):
        print(f"{i+1}th feature is {ff} and it is : ")
        print(dataset["train"][0][ff])


# if i need to visualize data --> vis_data(dataset)

data = dataset["train"]
ds_with_embeddings = data.map(
    lambda example: {
        "embeddings": model.get_text_features(
            **tokenizer([example["image_description"]], truncation=True, return_tensors="pt").to("cuda")
        )[0]
        .detach()
        .cpu()
        .numpy()
    }
)

ds_with_embeddings = ds_with_embeddings.map(
    lambda example: {
        "image_embeddings": model.get_image_features(**processor([example["image"]], return_tensors="pt").to("cuda"))[
            0
        ]
        .detach()
        .cpu()
        .numpy()
    }
)
