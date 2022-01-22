import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, nbr_features, num_layers=3, hidden_size=10, pdrop=0.1):
        super().__init__()
        
        self.pdrop = pdrop
        
        self.fc1 = nn.Linear(nbr_features, hidden_size)
        
        self.hidden = nn.ModuleList()
        for k in range(num_layers - 1):
            self.hidden.append(nn.Linear(hidden_size, hidden_size))
        
        self.fc4 = nn.Linear(hidden_size, 1)
        
        self.bn = nn.BatchNorm1d(hidden_size, affine=False)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, x):
        
        x = x.flatten(start_dim=1)
        x = F.relu(self.dropout(self.fc1(x)))
        for layer in self.hidden:
            x = F.relu(self.dropout(layer(x)))
        
        x = torch.sigmoid(self.fc4(x))       

        return x.squeeze()


class RNN(nn.Module):

    def __init__(self, nbr_features, hidden_size, num_layers=1, pdrop=0.):
        super(RNN, self).__init__()
        
        self.nbr_features = nbr_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = pdrop
        self.rnn = nn.RNN(input_size=self.nbr_features, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), x.size(2))
        x, _ = self.rnn(x)
        x = self.sigmoid(self.fc(x[:, -1, :]))

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