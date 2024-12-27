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

    def format(example):
        
        prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": example["question"]}]}]
        chosen = [{"role": "assistant", "content": [{"type": "text", "text": example["chosen"]}]}]
        rejected = [{"role": "assistant", "content": [{"type": "text", "text": example["rejected"]}]}]
        prompt = processor.apply_chat_template(prompt, tokenize=False)
        chosen = processor.apply_chat_template(chosen, tokenize=False)
        rejected = processor.apply_chat_template(rejected, tokenize=False)
        max_size = processor.image_processor.size["longest_edge"] // 2
        example["image"].thumbnail((max_size, max_size))
        return {"images": [example["image"]], "prompt": prompt, "chosen": chosen, "rejected": rejected}

    dataset = dataset.map(format, remove_columns=dataset.column_names, num_proc=32)
    f = dataset.features
    f["images"] = features.Sequence(features.Image(decode=True))
    dataset = dataset.cast(f)

    print("Setting up training...")
    training_args = DPOConfig(
        output_dir="idefics2-8b-dpo",
        bf16=True,
        gradient_checkpointing=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=1,
        dataset_num_proc=32,
        dataloader_num_workers=32,
        logging_steps=10,
    )
    trainer = DPOTrainer(
        model,
        ref_model=None,  # Not needed when using PEFT
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        peft_config=LoraConfig(target_modules="all-linear"),
    )
    trainer.train()

if __name__ == "__main__":
    main()