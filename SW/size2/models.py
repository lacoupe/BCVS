import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, nbr_features, pdrop=0.2, hidden_size1=20, hidden_size2=20, hidden_size3=20):
        super().__init__()

        self.fc1 = nn.Linear(nbr_features, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, 2)
        self.drop = nn.Dropout(pdrop)
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.bn1(self.fc1(x))))
        x = self.relu(self.drop(self.bn2(self.fc2(x))))
        x = self.relu(self.drop(self.fc3(x)))
        x = self.softmax(self.fc4(x))       

        return x