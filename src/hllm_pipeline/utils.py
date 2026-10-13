import json
from pathlib import Path


def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def write_file(path, text, overwrite=True):
    path = Path(path)
    if not overwrite and path.exists():
        raise FileExistsError(f"Me niego a sobrescribir un archivo existente: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)

def read_json(path):
    return json.loads(read_text_file(path))

def write_json(path, data, overwrite=True):
    write_file(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n", overwrite=overwrite)
