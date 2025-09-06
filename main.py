import torch
from transformers import  T5Tokenizer 

from data import build_dataset
from  log import log 
from soft_prompt_tuning import soft_prompt_tuning 
from adapter_tuning import adapter_tuning
from lora import lora

# Config 

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")

model_name = "t5-small"

# Tokenizer setup
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load dataset splits
train_dataset, test_dataset, eval_dataset = build_dataset(tokenizer)

# # Run all fine-tuning strategies
# soft_prompt_tuning(tokenizer, device, train_dataset, eval_dataset, test_dataset)
# adapter_tuning(tokenizer,train_dataset, test_dataset, eval_dataset)
lora(tokenizer, train_dataset, test_dataset, eval_dataset, device)

