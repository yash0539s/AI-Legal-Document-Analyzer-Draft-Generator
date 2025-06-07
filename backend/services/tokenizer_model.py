from transformers import GPT2Tokenizer, GPT2LMHeadModel
import os

model_name = "gpt2"  # You can change this to "gpt2-medium", "gpt2-large", etc.
save_dir = "artifacts/gpt2_model"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

print(f"Downloading and saving tokenizer and model to '{save_dir}' ...")

# Download and save tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

# Download and save model
model = GPT2LMHeadModel.from_pretrained(model_name)
model.save_pretrained(save_dir)

print("Download and save complete!")