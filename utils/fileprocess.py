import json


def read_json(path):
    with open(path, "rb") as f:
        return json.load(f)


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, skipkeys=True)


def read_file(path):
    data=[]
    with open(path, "r") as f:
        for line in f:
            data.append(line.strip())
    return data