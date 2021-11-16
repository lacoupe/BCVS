import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, x1, x2, x3, pdrop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(x1 * x2 * x3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 4)
        self.drop = nn.Dropout(pdrop)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x

    
class ConvNet(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim1_kernel1=2, dim2_kernel1=2, dim1_kernel2=2, dim2_kernel2=2, pdrop=0.2):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(dim1_kernel1, dim2_kernel1, dim3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(dim1_kernel2, dim2_kernel2, 1))
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16 * (dim1 - dim1_kernel1 - dim1_kernel2 + 2) * (dim2 - dim2_kernel1 - dim2_kernel2 + 2), 32)
        self.fc2 = nn.Linear(32, 4)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(pdrop)
        self.drop2d = nn.Dropout2d(pdrop)

    def forward(self, x):
        
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
    
        x = self.relu(self.drop2d(self.conv1(x)))
        x = self.relu(self.drop2d(self.conv2(x)))
        
        #print(x.shape)
        x = x.flatten(start_dim=1)

        x = self.relu(self.drop(self.fc1(x)))
        # x = self.relu(self.drop(self.fc2(x)))
        x = self.sigmoid(self.fc2(x))

        return x
    
    
class LSTM(nn.Module):

    def __init__(self, input_size, output_size, hidden_size=20, num_layers=5, dropout=0.1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x, _ = self.lstm(x, (h0, c0))
        x = self.relu(x)
        x = self.sigmoid(self.fc(x[:, -1, :]))
        
        return x
   