from data import get_price_data
from test import get_training_processed_data
from train_test import train, output_to_loss, output_to_accu
from models import MLP, ConvNet, LSTM
from sklearn.model_selection import train_test_split
import torch

def test_model():

    # Get data
    price, _, df_input_all = get_price_data()

    # Data parameters
    rebalance_freq = 'W-FRI'
    input_period = 42
    training_window = 8

    # Process data
    X, y = get_training_processed_data(df_input_all, price, rebalance_freq, input_period, training_window)
    train_indices, test_indices, _, _ = train_test_split(range(len(y)), y, stratify=y, test_size=0.3)
    X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]

    mean = X_train.mean(dim=[0, 1, 2], keepdim=True)
    std = X_train.std(dim=[0, 1, 2], keepdim=True)
    X_train = X_train.sub_(mean).div_(std)
    X_test = X_test.sub_(mean).div_(std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    eta_mlp = 1e-3
    eta_convnet = 1e-3
    eta_lstm = 1e-3
    weight_decay = 1e-5
    dropout = 0.2
    nb_epochs = 100
    batch_size = 5
    verbose = 2
    i = 0
    
    model_name = 'LSTM'
    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)
    if model_name == 'MLP':
        eta = eta_mlp
        model = MLP(dim1, dim2, dim3, dropout)
    elif model_name == 'ConvNet':
        eta = eta_convnet
        model = ConvNet(dim1, dim2, dim3, dropout)
    elif model_name == 'LSTM':
        eta = eta_lstm
        model = LSTM(input_size=dim2 * dim3, output_size=dim2, device=device, pdrop=dropout)

    train(model, X_train, y_train, nb_epochs=nb_epochs, device=device, X_test=X_test, y_test=y_test, i=i, batch_size=batch_size, eta=eta, 
        weight_decay=weight_decay, verbose=verbose, classification=True)

    print(f'Accuracy on train set :{output_to_accu(model, X_train, y_train):.2f}')
    print(f'Accuracy on train set :{output_to_accu(model, X_test, y_test):.2f}')

if __name__ == "__main__":
    test_model()