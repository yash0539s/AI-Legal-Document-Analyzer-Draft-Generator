import json
import os
import re

def clean_text(text):
    if not text:
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  # normalize whitespace
    text = text.strip()
    return text

def preprocess_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ids = data.get('id', [])
    answers = data.get('answers', [])

    processed = []
    for i, id_val in enumerate(ids):
        if i < len(answers):
            ans_obj = answers[i]
            if 'text' in ans_obj and ans_obj['text']:
                raw_text = ans_obj['text'][0]
                cleaned_text = clean_text(raw_text)
                answer_start = ans_obj['answer_start'][0] if 'answer_start' in ans_obj and ans_obj['answer_start'] else None
            else:
                cleaned_text = None
                answer_start = None
        else:
            cleaned_text = None
            answer_start = None

        processed.append({
            'id': id_val,
            'answer_text': cleaned_text,
            'answer_start': answer_start
        })
    return processed

def write_processed_to_file(processed_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(f"ID: {item['id']}\n")
            f.write(f"Answer Text: {item['answer_text']}\n")
            f.write(f"Answer Start: {item['answer_start']}\n\n")
    print(f"Saved processed data to {output_path}")

# Paths to your JSON files
train_json_path = 'da/train.json'  # change if needed
test_json_path = 'da/test.json'    # change if needed

# Output files
train_output_path = 'preprocess/train/doctxt.txt'
test_output_path = 'preprocess/test/doctxt.txt'

# Preprocess and save
train_processed = preprocess_json(train_json_path)
write_processed_to_file(train_processed, train_output_path)

test_processed = preprocess_json(test_json_path)
write_processed_to_file(test_processed, test_output_path)
