from transformers import GPT2LMHeadModel, GPT2Tokenizer
from backend.core.config import config

class DraftGenerator:
    def __init__(self):
        try:
            # Access path to draft model directory from config
            model_path = config.data.paths.artifacts.draft_model_dir
        except AttributeError:
            raise ValueError("Draft model path not found in config")

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.model.eval()

    def generate(self, prompt: str, max_length=100):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')

        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id  # fix warning about pad token
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
