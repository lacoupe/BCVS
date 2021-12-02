from data import get_price_data
from test import get_training_processed_data
from train_test import train, output_to_loss, output_to_accu
from models import MLP
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torch

def tune_model():

    # Get data
    price, _, df_input_all = get_price_data()

    # Data parameters
    rebalance_freq = 'W-FRI'
    input_period = 42
    training_window = 6

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
    
    # Grid parameters
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    weight_decays = [1e-4, 5e-4, 1e-3, 5e-3]
    dropouts = [0.1, 0.20, 0.25, 0.30]
    nb_epochs = [100, 200, 300]

    # Fixed ML parameters
    # nb_epochs = 100
    batch_size = 5

    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)

    tuning_list = []
    for epoch in tqdm(nb_epochs, leave=False):
        for lr in tqdm(learning_rates, leave=False):
            for w in tqdm(weight_decays, leave=False):
                for drop in tqdm(dropouts, leave=False):
                    model = MLP(dim1, dim2, dim3, pdrop=drop).to(device)
                    train(model, X_train, y_train, epoch, batch_size=batch_size, eta=lr, weight_decay=w, verbose=0)
                    tuning_list.append([lr, w, drop, epoch, np.round(output_to_loss(model, X_test, y_test).item(), 2), np.round(output_to_accu(model, X_test, y_test), 2)])
    print(pd.DataFrame(data=tuning_list, columns=['Learning_rate', 'Weight_decay', 'Dropout', 'Nb_epochs', 'Loss', 'Accuracy']).sort_values('Loss').to_string(index=False))

if __name__ == "__main__":
    tune_model()