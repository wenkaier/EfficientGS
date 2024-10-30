import json

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as fr:
        data = json.load(fr)
    return data

def save_json(filename: str, data, indent: int = 4):
    with open(filename, "w", encoding="utf-8") as fw:
        json.dump(data, fw, ensure_ascii=False, indent=indent)
