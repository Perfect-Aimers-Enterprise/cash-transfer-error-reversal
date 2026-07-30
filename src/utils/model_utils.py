import torch

from src.models.gru import GRUClassifier


def save_model(
    model,
    model_path,
    input_size,
    hidden_size,
    num_layers,
    num_classes,
    classes,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "classes": classes,
    }

    torch.save(
        checkpoint,
        model_path,
    )

    print(f"\nModel saved successfully:\n{model_path}")


def load_model(model_path, device):

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model = GRUClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model, checkpoint