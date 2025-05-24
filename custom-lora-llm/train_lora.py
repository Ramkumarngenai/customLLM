# train_lora.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch

model_name = "meta-llama/Llama-3-8B"  # Use a smaller model if needed (e.g., "llama-3b")

# Load your dataset
dataset = load_dataset("json", data_files="customer_service.jsonl", split="train")

# Prepare tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    input_text = f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['completion']}"
    tokenized = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    return tokenized

tokenized_dataset = dataset.map(tokenize)

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Based on model internals
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training config
training_args = TrainingArguments(
    output_dir="./customer-service-lora",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# Train
trainer.train()
model.save_pretrained("./customer-service-lora")