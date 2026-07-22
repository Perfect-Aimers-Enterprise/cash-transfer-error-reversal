import json


def save_metadata(path, input_size, classes):

    metadata = {
        "input_size": input_size,
        "classes": classes,
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)