import torch.nn as nn


class TransactionClassifier(nn.Module):

    def __init__(self, input_size, num_classes):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),

            nn.ReLU(),

            nn.Linear(64, 32),

            nn.ReLU(),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):

        return self.network(x)