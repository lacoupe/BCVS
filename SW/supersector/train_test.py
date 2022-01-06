import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os
import torch.nn.functional as F
np.set_printoptions(precision=3, suppress=True)


def output_to_accu(model, X, y):
    model.eval()
    prob = test(model, X)
    pred = (prob >= 0.5).astype(int)
    nb_errors = 0
    for b in range(0, X.size(0)):
        if pred[b] != y[b]:
            nb_errors = nb_errors + 1
    accuracy = 100 * (1 - nb_errors / X.size(0))
    return accuracy


def output_to_loss(model, X, y):
    model.eval()
    loss = 0
    criterion = nn.BCELoss()
    for b in range(0, X.size(0)):
        output = model(X.narrow(0, b, 1))
        loss += criterion(output, y[b])
    return (loss / X.size(0)).cpu()


def test(model, X_test):
    model.eval()
    prob = model(X_test).cpu().detach().numpy()
    return prob


def train(model, X_train, y_train, nb_epochs, X_test=None, y_test=None, i=None, eta=1e-3, weight_decay=0, batch_size=1, verbose=0):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)

    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()

    model.train()
    
    if verbose in (1, 2):
        train_accu_list = []
        train_loss_list = []
        test_accu_list = []
        test_loss_list = []
    
    train_set = TensorDataset(X_train, y_train)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    for e in (tqdm(range(nb_epochs)) if (verbose == 3) else range(nb_epochs)):
        acc_loss = 0
        model.train()
        for train_input, train_target in train_loader:
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(output, train_target)
            loss.backward()
            # clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            if verbose in (2, 4):
                acc_loss = acc_loss + loss.item()

        if verbose in (2, 4):
            print('epoch', e + 1, 'loss :', np.round(acc_loss, 4), 'accuracy :', np.round(output_to_accu(model, X_train, y_train), 2), '%')

        if verbose in (1, 2):
            model.eval()
            train_accu = output_to_accu(model, X_train, y_train)
            train_accu_list.append(train_accu)
            test_accu = output_to_accu(model, X_test, y_test)
            test_accu_list.append(test_accu)
            
            train_loss = output_to_loss(model, X_train, y_train).detach().numpy()
            train_loss_list.append(train_loss)
            test_loss = output_to_loss(model, X_test, y_test).detach().numpy()
            test_loss_list.append(test_loss)
                    
    if verbose in (1, 2):
        _, axs = plt.subplots(2, 1, figsize=(12,8))
        axs[0].plot(list(range(nb_epochs)), train_loss_list, label='Train loss')
        axs[0].plot(list(range(nb_epochs)), test_loss_list, label='Test loss')
        axs[0].legend()
        axs[1].plot(list(range(nb_epochs)), train_accu_list, label='Train accuracy')
        axs[1].plot(list(range(nb_epochs)), test_accu_list, label='Test accuracy')
        axs[1].legend()
        plt.xlabel('Epoch')
        plt.suptitle('Learning Curve ' + model.__class__.__name__, fontsize=15)
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/learning_curve_' + model.__class__.__name__ + '.png'
        plt.savefig(plot_path)
        plt.show()



    
    
