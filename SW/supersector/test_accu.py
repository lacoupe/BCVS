from data import get_data, get_processed_data
from train_test import train, output_to_accu
from helpers import plot_cm, count_parameters
from models import MLP
from sklearn.model_selection import train_test_split
import torch
import numpy as np

def test_model():

    # Get data
    _, target_prices, features = get_data()

    training_window = 10
    X, y = get_processed_data(features, target_prices, training_window)

    # print(X[265:280])

    train_indices, test_indices, _, _ = train_test_split(range(len(y)), y, test_size=0.2, shuffle=False, random_state=1)
    X_train, y_train, X_test, y_test = X[train_indices], y[train_indices], X[test_indices], y[test_indices]
    print('Number of train sample', len(X_train))
    print('Number of test sample', len(X_test))
    X_mean = X_train.mean(dim=[0], keepdim=True)
    X_std = X_train.std(dim=[0], keepdim=True)
    X_train = X_train.sub_(X_mean).div_(X_std)
    X_test = X_test.sub_(X_mean).div_(X_std)

    class_count_train = np.unique(y_train.cpu(), axis=0, return_counts=True)[1]
    class_count_test = np.unique(y_test.cpu(), axis=0, return_counts=True)[1]
    weights_train = torch.tensor(class_count_train / sum(class_count_train))
    weights_test = torch.tensor(class_count_test / sum(class_count_test))
    print('Allocation of best returns in train set :', weights_train.cpu().numpy())
    print('Allocation of best returns in test set :', weights_test.cpu().numpy())
    
    eta = 1e-3
    weight_decay = 1e-4
    dropout = 0.1
    batch_size = 20
    print(f'parameters of ML model : \nlearning_rate = {eta}, weight_decay = {weight_decay}, dropout = {dropout}, batch_size = {batch_size}')

    nb_epochs = 30

    verbose = 2

    nbr_features = X.size(1)

    model = MLP(nbr_features, pdrop=dropout)

    print(f"Number of {model.__class__.__name__}'s parameters : {count_parameters(model)}")
    print('number of feature :', nbr_features)

    train(model, X_train, y_train, nb_epochs=nb_epochs, X_test=X_test, y_test=y_test, batch_size=batch_size, eta=eta, 
          weight_decay=weight_decay, verbose=verbose)

    print(f'Accuracy on train set : {output_to_accu(model, X_train, y_train):.2f} %')
    print(f'Accuracy on test set : {output_to_accu(model, X_test, y_test):.2f} %')
    plot_cm(model, X_test, y_test, target_prices)

if __name__ == "__main__":
    test_model()