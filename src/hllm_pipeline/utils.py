import json


def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def write_file(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def write_json(path, data):
    write_file(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
