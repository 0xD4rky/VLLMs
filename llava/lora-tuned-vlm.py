from datasets import features, load_dataset
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
import torch.quantization as tq

def main():

    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b", do_image_splitting=False)

    print("Loading dataset...")
    dataset = load_dataset("openbmb/RLAIF-V-Dataset", split="train")

    model = tq.quantize_dynamic(
        model, 
        {torch.nn.Linear},  
        dtype=torch.qint8  
    )