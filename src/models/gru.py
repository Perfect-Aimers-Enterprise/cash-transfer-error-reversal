import torch.nn as nn


class GRUClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(
            hidden_size,
            num_classes,
        )

    def forward(self, x):

        output, hidden = self.gru(x)

        # Last hidden state
        x = hidden[-1]

        x = self.fc(x)

        return x