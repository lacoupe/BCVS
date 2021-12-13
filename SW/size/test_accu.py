from torch.utils.data.dataset import random_split
from data import get_price_data
from test import get_training_processed_data
from train_test import train, output_to_loss, output_to_accu, test, train_siamese
from helpers import plot_cm
from models import MLP, ConvNet, LSTM, SiameseLSTM, SiameseConvNet
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def test_model():

    # Get data
    _, _, df_input_all, target_prices  = get_price_data()

    # Data parameters
    rebalance_freq = 'W-FRI'
    input_period = 42
    input_period_weeks = 8
    training_window = 10

    # Process data
    X, X_reg, y, y_reg = get_training_processed_data(df_input_all, target_prices, rebalance_freq, input_period, input_period_weeks, training_window)
    train_indices, test_indices, _, _ = train_test_split(range(len(y)), y, stratify=y, test_size=0.4, random_state=1)
    X_train, X_train_reg, y_train, y_train_reg, X_test, X_test_reg, y_test = X[train_indices], X_reg[train_indices], y[train_indices], y_reg[train_indices], X[test_indices], X_reg[test_indices], y[test_indices]

    X_mean = X_train.mean(dim=[0, 1, 2], keepdim=True)
    X_std = X_train.std(dim=[0, 1, 2], keepdim=True)
    X_train = X_train.sub_(X_mean).div_(X_std)
    X_test = X_test.sub_(X_mean).div_(X_std)

    X_reg_mean = X_train_reg.mean(dim=[0, 1], keepdim=True)
    X_reg_std = X_train_reg.std(dim=[0, 1], keepdim=True)
    X_train_reg = X_train_reg.sub_(X_reg_mean).div_(X_reg_std)

    y_reg_mean = y_train_reg.mean(dim=[0, 1], keepdim=True)
    y_reg_std = y_train_reg.std(dim=[0, 1], keepdim=True)
    y_train_reg = y_train_reg.sub_(y_reg_mean).div_(y_reg_std)

    

    class_count = np.unique(y_train.cpu(), axis=0, return_counts=True)[1]
    weights = torch.tensor(class_count / sum(class_count))
    print('Allocation of best returns', weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Number of train sample', len(X_train))
    print('Number of test sample', len(X_test))
    X_train = X_train.to(device)
    X_train_reg = X_train_reg.to(device)
    y_train = y_train.to(device)
    y_train_reg = y_train_reg.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    eta_mlp = 1e-3
    eta_convnet = 1e-3
    eta_lstm = 1e-3
    eta_lstm_siam = 1e-3
    eta_conv_siam = 1e-4

    weight_decay = 1e-5
    dropout = 0.
    nb_epochs = 500
    batch_size = 10
    verbose = 2
    gamma = 0.8
    
    model_name = 'SiameseLSTM'
    # model_name = 'ConvNet'
    siamese = True

    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)
    if model_name == 'MLP':
        eta = eta_mlp
        model = MLP(dim1, dim2, dim3, pdrop=dropout)
    elif model_name == 'ConvNet':
        eta = eta_convnet
        model = ConvNet(dim1, dim2, dim3, pdrop=dropout)
    elif model_name == 'LSTM':
        eta = eta_lstm
        model = LSTM(input_size=dim2 * dim3, output_size=dim2, device=device, pdrop=dropout)
    elif model_name == 'SiameseLSTM':
        eta = eta_lstm_siam
        model = SiameseLSTM(input_size=dim2 * dim3, output_size=X_reg.size(2), device=device, pdrop=dropout)
    elif model_name == 'SiameseConvNet':
        eta = eta_conv_siam
        model = SiameseConvNet(dim1, dim3, pdrop=dropout)

    model.to(device)
    if siamese:
        train_siamese(model, X_train, X_train_reg, y_train, y_train_reg=y_train_reg, nb_epochs=nb_epochs, device=device, X_test=X_test, y_test=y_test, batch_size=batch_size, eta=eta, 
            weight_decay=weight_decay, verbose=verbose, gamma=gamma)
    else:
        train(model, X_train, y_train, nb_epochs=nb_epochs, device=device, X_test=X_test, y_test=y_test, batch_size=batch_size, eta=eta, 
            weight_decay=weight_decay, verbose=verbose, classification=True)

    print(f'Accuracy on train set : {output_to_accu(model, X_train, X_train_reg, y_train):.2f} %')
    print(f'Accuracy on test set : {output_to_accu(model, X_test, X_test_reg, y_test):.2f} %')
    plot_cm(model, X_test, X_test_reg, y_test, target_prices)


if __name__ == "__main__":
    test_model()