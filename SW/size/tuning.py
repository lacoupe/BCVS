from train_test import train, output_to_loss, output_to_accu
from models import MLP
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from data import get_data, get_processed_data
import torch

def tune_model():

    # Get data
    _, target_prices, _, features = get_data()

    input_period = 42
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device available :', device)
    
    X_train = X_train.to(device)
    X_train_reg = X_train_reg.to(device)
    y_train = y_train.to(device)
    y_train_reg = y_train_reg.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Grid parameters
    learning_rates = [1e-4, 1e-3, 1e-2]
    weight_decays = [1e-4, 1e-3, 1e-2]
    dropouts = [0.1, 0.2, 0.3]
    batch_sizes = [20, 40, 60]

    # Fixed ML parameters
    nb_epochs = 30

    dim1, dim2 = X.size(1), X.size(2)

    tuning_list = []
    for b in tqdm(batch_sizes, leave=False, position=1):
        for lr in tqdm(learning_rates, leave=False, position=1):
            for w in tqdm(weight_decays, leave=False, position=2):
                for drop in tqdm(dropouts, leave=False, position=3):
                    model = MLP(dim1, dim2, pdrop=drop).to(device)
                    train(model, X_train, y_train, nb_epochs, device=device, batch_size=b, eta=lr, weight_decay=w, verbose=0)
                    tuning_list.append([lr, w, drop, nb_epochs, np.round(output_to_loss(model, X_test, y_test).item(), 2), np.round(output_to_accu(model, X_test, y_test), 2)])
    print(pd.DataFrame(data=tuning_list, columns=['Learning_rate', 'Weight_decay', 'Dropout', 'Nb_epochs', 'Loss', 'Accuracy']).sort_values('Accuracy').to_string(index=False))

if __name__ == "__main__":
    tune_model()