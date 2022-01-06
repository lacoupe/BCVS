import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, nbr_features, pdrop=0.15, hidden_size=10):
        super().__init__()

        self.fc1 = nn.Linear(nbr_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.drop1 = nn.Dropout(pdrop)
        self.drop2 = nn.Dropout(pdrop)
        self.drop3 = nn.Dropout(pdrop)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
        
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu1(self.bn1(self.drop1(self.fc1(x))))
        x = self.relu2(self.bn2(self.drop2(self.fc2(x))))
        x = self.relu3(self.drop3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))       

        return x.squeeze()


class GRU(nn.Module):

    def __init__(self, nbr_features, hidden_size, num_layers=1, pdrop=0.):
        super(GRU, self).__init__()
        
        self.nbr_features = nbr_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = pdrop
        self.gru = nn.GRU(input_size=self.nbr_features, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2))
        x, _ = self.gru(x)
        x = self.sigmoid(self.fc(x[:, -1, :]))

        return x.squeeze()


class LSTM(nn.Module):

    def __init__(self, nbr_features, hidden_size, num_layers=1, pdrop=0.):
        super().__init__()
        self.nbr_features = nbr_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = pdrop
        self.lstm = nn.LSTM(input_size=self.nbr_features, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2))
        x, _ = self.lstm(x)
        x = self.sigmoid(self.fc(x[:, -1, :]))

        return x.squeeze()