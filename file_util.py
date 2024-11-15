import json


def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None


def write_json(data, file_path, mode="w"):
    try:
        with open(file_path, mode) as file:
            json.dump(data, file, indent=4)
    except IOError:
        print(f"Error writing to file: {file_path}")
