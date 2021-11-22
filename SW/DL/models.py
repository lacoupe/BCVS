import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, pdrop=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim1 * dim2 * dim3, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, dim2)
        self.drop = nn.Dropout(pdrop)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.relu(self.drop(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        
        return x

    
class ConvNet(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim1_kernel1=2, dim2_kernel1=2, dim1_kernel2=2, dim2_kernel2=2, pdrop=0.3):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
        self.conv1 = nn.Conv3d(1, 4, kernel_size=(dim1_kernel1, dim2_kernel1, dim3))
        self.conv2 = nn.Conv3d(4, 8, kernel_size=(dim1_kernel2, dim2_kernel2, 1))
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(8 * (dim1 - dim1_kernel1 - dim1_kernel2 + 2) * (dim2 - dim2_kernel1 - dim2_kernel2 + 2), 10)
        self.fc2 = nn.Linear(10, self.dim2)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.drop = nn.Dropout(pdrop)
        self.drop2d = nn.Dropout2d(pdrop)

    def forward(self, x):
        
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
    
        x = self.relu(self.drop2d(self.conv1(x)))
        x = self.relu(self.drop2d(self.conv2(x)))
        
        # print(x.shape)
        x = x.flatten(start_dim=1)

        x = self.relu(self.drop(self.fc1(x)))
        # x = self.relu(self.drop(self.fc2(x)))
        x = self.softmax(self.fc2(x))

        return x
    
    
class LSTM(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_size=20, num_layers=5, dropout=0.1):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.relu(x)
        x = self.softmax(self.fc(x[:, -1, :]))
        
        return x
   