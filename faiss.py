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

ds_with_embeddings.add_faiss_index(column="embeddings")
ds_with_embeddings.add_faiss_index(column="image_embeddings")

def search(query, k=3):
    """
    Search for the k most similar examples to the query using the FAISS index
    and return the scores and the retrieved examples.

    args: 
    1. query (str): the query to search for
    2. k (int): the number of examples to retrieve
    """
    prompt_embedding = (
    model.get_text_features(
        **tokenizer([query], return_tensors = "pt", truncation = True
    ).to("cuda"))[0]
    .detach()
    .cpu()
    .numpy()
    )

    scores, retrieved_examples = ds_with_embeddings.get_nearest_examples("embeddings", prompt_embedding, k=k)
    return scores, retrieved_examples

scores, retrieved_examples = search("snow on ground")
print(f"the length of train dataset: {len(dataset['train'])}")
print(f"the length of query retrieved set: {len(retrieved_examples)}")

def downscale_image(image):

  """
  downscaling the image for better view
  """
  width = 200
  ratio = width / float(image.size[0])
  height = int(image.size[1] * float(ratio))
  img = image.resize((width,height),Image.Resampling.LANCZOS)
  return img

images = [downscale_image(image) for image in retrieved_examples["image"]]
print(retrieved_examples["image_description"][2])
images[2]