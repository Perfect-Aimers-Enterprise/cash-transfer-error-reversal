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
            nn.Dropout(0.2),

            nn.Linear(
                hidden_size,
                32,
            ),

            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(
                32,
                2,
            ),

        )

    def forward(self, x):

        return self.network(x)