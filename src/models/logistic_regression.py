import torch
import torch.nn as nn

class LogisticRegressionClassifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.linear = nn.Linear(
            input_size,
            2,
        )

    def forward(self, x):

        return self.linear(x)