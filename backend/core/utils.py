import re
import string

def clean_text(text: str, lowercase=True, remove_punct=True) -> str:
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    # remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
