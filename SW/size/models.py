import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim1, dim2, dim3, pdrop=0.2, hidden_size1=20, hidden_size2=15, hidden_size3=10):
        super().__init__()

        self.fc1 = nn.Linear(dim1 * dim2 * dim3, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, dim2)
        self.drop = nn.Dropout(pdrop)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.bn1(self.fc1(x))))
        x = self.relu(self.drop(self.bn2(self.fc2(x))))
        x = self.relu(self.drop(self.fc3(x)))
        x = self.softmax(self.fc4(x))       

        return x, None

    
class ConvNet(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim1_kernel1=10, dim2_kernel1=2, dim1_kernel2=5, dim2_kernel2=2, pdrop=0.3):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(dim1_kernel1, dim2_kernel1, dim3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(dim1_kernel2, dim2_kernel2, 1))
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16 * (dim1 - dim1_kernel1 - dim1_kernel2 + 2) * (dim2 - dim2_kernel1 - dim2_kernel2 + 2), 10)
        self.fc2 = nn.Linear(10, self.dim2)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(pdrop)
        self.drop3d = nn.Dropout3d(pdrop)
        self.bn3d1 = nn.BatchNorm3d(8)
        self.bn3d2 = nn.BatchNorm3d(16)

    def forward(self, x):
        
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        # print(x.size())
        x = self.relu(self.drop3d(self.bn3d1(self.conv1(x))))
        x = self.relu(self.drop3d(self.bn3d2(self.conv2(x))))
    
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.softmax(self.fc2(x))

        return x, None
    
    
class LSTM(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_size=8, num_layers=2, pdrop=0.2):
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = pdrop
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.softmax(self.fc(x[:, -1, :]))

        return x, None


class SiameseConvNet(nn.Module):
    def __init__(self, dim1, dim2, dim1_kernel1=10, dim2_kernel1=3, dim1_kernel2=5, dim2_kernel2=1, pdrop=0.3):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(dim1_kernel1, dim2_kernel1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(dim1_kernel2, dim2_kernel2))

        self.fc1 = nn.Linear(16 * (dim1 - (dim1_kernel1 -1) - (dim1_kernel2 - 1)) * (dim2 - (dim2_kernel1 - 1) - (dim2_kernel2 - 1)), 10)
        self.fc2 = nn.Linear(10, self.dim2)
        self.fc3 = nn.Linear(3, 3)
        self.fc4 = nn.Linear(3, 3)
        self.fc_once = nn.Linear(10, 1)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.drop = nn.Dropout(pdrop)
        self.drop2d = nn.Dropout2d(pdrop)
        self.bn2d1 = nn.BatchNorm2d(8)
        self.bn2d2 = nn.BatchNorm2d(16)

    def forward_once(self, x):
    
        x = self.relu(self.drop2d(self.bn2d1(self.conv1(x))))
        x = self.relu(self.drop2d(self.bn2d2(self.conv2(x))))
    
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.fc_once(x)

        return x
    
    def forward(self, x):

        input1 = x[:, :, 0, :].view(x.size(0), 1, x.size(1), x.size(3))
        input2 = x[:, :, 1, :].view(x.size(0), 1, x.size(1), x.size(3))
        input3 = x[:, :, 2, :].view(x.size(0), 1, x.size(1), x.size(3))

        x1 = self.forward_once(input1)
        x2 = self.forward_once(input2)
        x3 = self.forward_once(input3)

        auxiliary = torch.cat((x1, x2, x3), 1)

        output = self.relu(self.fc3(auxiliary))
        output = self.softmax(self.fc4(output))

        return output, auxiliary



class SiameseLSTM(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_size=10, num_layers=1, pdrop=0.1):
        super().__init__()
        
        self.device = device

        # Input data size
        self.input_size = input_size
        self.output_size = output_size

        # LSTM parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Regularizers
        self.dropout = pdrop

        # Regression
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc_once = nn.Linear(hidden_size, 1)

        # Classification
        self.lstm2 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                             num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)

        # Final branch
        self.fc_last = nn.Linear(6, 6)
        self.classifier = nn.Linear(6, 3)

        # Activation function
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward_once(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm1(x, (h0.detach(), c0.detach()))
        x = self.fc_once(x[:, -1, :])
        return x

    def forward(self, x):

        # Forward Regression with weight sharing
        input1 = x[:, :, 0, -1].view(x.size(0), x.size(1), 1).to(self.device)
        input2 = x[:, :, 1, -1].view(x.size(0), x.size(1), 1).to(self.device)
        input3 = x[:, :, 2, -1].view(x.size(0), x.size(1), 1).to(self.device)
        
        x1 = self.forward_once(input1)
        x2 = self.forward_once(input2)
        x3 = self.forward_once(input3)

        auxiliary = torch.cat((x1, x2, x3), 1)

        # Forward Classification
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        output = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        output, _ = self.lstm2(output, (h0.detach(), c0.detach()))
        output = self.softmax(self.fc(output[:, -1, :]))

        output = torch.cat((auxiliary, output), 1)
        output = self.relu(self.fc_last(output))
        output = self.softmax(self.classifier(output))

        return output, auxiliary









class MLP_REG(nn.Module):
    def __init__(self, dim1, dim2, dim3, pdrop=0.2, hidden_size1=20, hidden_size2=15, hidden_size3=10):
        super().__init__()

        self.fc1 = nn.Linear(dim1 * dim2 * dim3, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, dim2)
        self.drop = nn.Dropout(pdrop)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.bn1(self.fc1(x))))
        x = self.relu(self.drop(self.bn2(self.fc2(x))))
        x = self.relu(self.drop(self.fc3(x)))
        x = self.fc4(x)   
         
        return x
   

class ConvNet_REG(nn.Module):
    def __init__(self, dim1, dim2, dim3, dim1_kernel1=3, dim2_kernel1=2, dim1_kernel2=2, dim2_kernel2=2, pdrop=0.3):
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(dim1_kernel1, dim2_kernel1, dim3))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(dim1_kernel2, dim2_kernel2, 1))
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        self.fc1 = nn.Linear(16 * (dim1 - dim1_kernel1 - dim1_kernel2 + 2) * (dim2 - dim2_kernel1 - dim2_kernel2 + 2), 10)
        self.fc2 = nn.Linear(10, self.dim2)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.drop3d = nn.Dropout3d(pdrop)
        self.bn3d1 = nn.BatchNorm3d(8)
        self.bn3d2 = nn.BatchNorm3d(16)

    def forward(self, x):
        
        x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
    
        x = self.relu(self.drop3d(self.bn3d1(self.conv1(x))))
        x = self.relu(self.drop3d(self.bn3d2(self.conv2(x))))
    
        x = x.flatten(start_dim=1)
        x = self.relu(self.drop(self.fc1(x)))
        x = self.fc2(x)

        return x


class LSTM_REG(nn.Module):

    def __init__(self, input_size, output_size, device, hidden_size=8, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
        x, _ = self.lstm(x, (h0.detach(), c0.detach()))
        x = self.relu(x)
        x = self.fc(x[:, -1, :])
        
        return x