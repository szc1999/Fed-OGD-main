import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, num_classes=10, input_size=784):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out