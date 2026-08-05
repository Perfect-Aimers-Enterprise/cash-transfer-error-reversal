import torch
import torch.nn as nn

class ANNClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                hidden_size,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                2,
            ),

        )

    def forward(self, x):

        return self.network(x)