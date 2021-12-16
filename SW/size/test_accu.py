from numpy.lib.twodim_base import triu_indices
from torch.utils.data.dataset import random_split
from data import get_data, get_processed_data
from train_test import train, output_to_loss, output_to_accu, test, train_siamese, output_to_accu_siamese
from helpers import plot_cm, count_parameters, plot_cm_siamese
from models import MLP, ConvNet, LSTM, SiameseLSTM
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def test_model():

    # Get data
    _, target_prices, _, features = get_data()

    # Data parameters
    input_period = 21
    input_period_reg = 10
    training_window = 10

    # Process data
    X, X_reg, y, y_reg = get_processed_data(features, target_prices, input_period, input_period_reg, training_window)

    train_indices, test_indices, _, _ = train_test_split(range(len(y)), y, test_size=0.2, shuffle=False, random_state=1)
    X_train, X_train_reg, y_train, y_train_reg, X_test, X_test_reg, y_test = X[train_indices], X_reg[train_indices], y[train_indices], y_reg[train_indices], X[test_indices], X_reg[test_indices], y[test_indices]
    print('Number of train sample', len(X_train))
    print('Number of test sample', len(X_test))

    X_mean = X_train.mean(dim=[0, 1], keepdim=True)
    X_std = X_train.std(dim=[0, 1], keepdim=True)
    X_train = X_train.sub_(X_mean).div_(X_std)
    X_test = X_test.sub_(X_mean).div_(X_std)

    X_reg_mean = X_train_reg.mean(dim=[0, 1], keepdim=True)
    X_reg_std = X_train_reg.std(dim=[0, 1], keepdim=True)
    X_train_reg = X_train_reg.sub_(X_reg_mean).div_(X_reg_std)

    y_reg_mean = y_train_reg.mean(dim=[0, 1], keepdim=True)
    y_reg_std = y_train_reg.std(dim=[0, 1], keepdim=True)
    y_train_reg = y_train_reg.sub_(y_reg_mean).div_(y_reg_std)

    class_count_train = np.unique(y_train.cpu(), axis=0, return_counts=True)[1]
    class_count_test = np.unique(y_test.cpu(), axis=0, return_counts=True)[1]
    weights_train = torch.tensor(class_count_train / sum(class_count_train))
    weights_test = torch.tensor(class_count_test / sum(class_count_test))
    print('Allocation of best returns in train set :', weights_train.cpu().numpy())
    print('Allocation of best returns in test set :', weights_test.cpu().numpy())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available :', device)
    
    X_train = X_train.to(device)
    X_train_reg = X_train_reg.to(device)
    y_train = y_train.to(device)
    y_train_reg = y_train_reg.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    eta = 1e-3
    weight_decay = 1e-4
    dropout = 0.2
    batch_size = 30
    gamma = 0.3
    print(f'parameters of ML model : \nlearning_rate = {eta}, weight_decay = {weight_decay}, dropout = {dropout}, batch_size = {batch_size}, gamma = {gamma}')

    nb_epochs = 40

    verbose = 2
    
    model_name = 'MLP'
    siamese = False

    dim1, dim2 = X.size(1), X.size(2)

    if model_name == 'MLP':
        model = MLP(dim1, dim2, pdrop=dropout)

    elif model_name == 'ConvNet':
        model = ConvNet(dim1, dim2, pdrop=dropout)

    elif model_name == 'LSTM':
        model = LSTM(input_size=dim2, output_size=y.size(1), device=device, pdrop=dropout)

    elif model_name == 'SiameseLSTM':
        model = SiameseLSTM(input_size=dim2, output_size=X_reg.size(2), device=device, pdrop=dropout)

    model.to(device)
    print(f"Number of {model.__class__.__name__}'s parameters : {count_parameters(model)}")
    print('number of feature :', dim2, 'lookback period :', dim1)

    if siamese:
        train_siamese(model, X_train, X_train_reg, y_train, y_train_reg=y_train_reg, nb_epochs=nb_epochs, device=device, X_test=X_test, y_test=y_test, batch_size=batch_size, eta=eta, 
            weight_decay=weight_decay, verbose=verbose, gamma=gamma)
    else:
        train(model, X_train, y_train, nb_epochs=nb_epochs, device=device, X_test=X_test, y_test=y_test, batch_size=batch_size, eta=eta, 
            weight_decay=weight_decay, verbose=verbose)

    if siamese:
        print(f'Accuracy on train set : {output_to_accu_siamese(model, X_train, X_train_reg, y_train):.2f} %')
        print(f'Accuracy on test set : {output_to_accu_siamese(model, X_test, X_test_reg, y_test):.2f} %')
        plot_cm_siamese(model, X_test, X_test_reg, y_test, target_prices)
    else:
        print(f'Accuracy on train set : {output_to_accu(model, X_train, y_train):.2f} %')
        print(f'Accuracy on test set : {output_to_accu(model, X_test, y_test):.2f} %')
        plot_cm(model, X_test, y_test, target_prices)

if __name__ == "__main__":
    test_model()