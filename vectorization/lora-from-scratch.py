import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x
    

"""
LoRA is a method that can be applied to various types of neural networks, 
not just generative models like GPT or image generation models. 
For this hands-on example, we will train a small BERT model for text classification 
because classification accuracy is easier to evaluate than generated text
"""

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)