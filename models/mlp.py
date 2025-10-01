import torch, torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))
        x = self.dropout(self.relu(self.fc4(x)))
        return self.fc5(x)

class CustomLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        assert reduction in ('mean','sum'); self.reduction = reduction
    def forward(self, outputs, targets):
        d = torch.sqrt(torch.sum((outputs - targets) ** 2, dim=1))
        return d.mean() if self.reduction == 'mean' else d.sum()
