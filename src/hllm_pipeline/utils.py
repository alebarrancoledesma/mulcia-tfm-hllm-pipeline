def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text
