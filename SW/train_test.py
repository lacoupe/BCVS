import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader

def output_to_accu(model, X, y):
    nb_errors = 0
    for b in range(0, X.size(0)):
        output = model(X.narrow(0, b, 1))
        _, predicted_classes = output.max(1)
        for k in range(1):
            if y[b + k, predicted_classes[k]] != 1.:
                nb_errors = nb_errors + 1
    accuracy = 100 * (1 - nb_errors / X.size(0))
    return accuracy


def output_to_loss(model, X, y):
    nb_errors = 0
    loss = 0
    criterion = nn.BCELoss()
    for b in range(0, X.size(0)):
        output = model(X.narrow(0, b, 1))
        loss += criterion(output, y.narrow(0, b, 1))
        
    return loss


def train(model, X_train, y_train, X_test, y_test, nb_epochs, i, eta=1e-3, batch_size=1, verbose=0):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    criterion = nn.BCELoss()
    model.train()
    
    if verbose in (1, 2):
        train_accu_list = []
        train_loss_list = []
        test_accu_list = []
        test_loss_list = []
    
    train_set = TensorDataset(X_train, y_train)    
    train_loader = DataLoader(train_set, batch_size=batch_size)
    
    for e in range(nb_epochs):
        acc_loss = 0
        for train_input, train_target in train_loader:
            optimizer.zero_grad()
            output = model(train_input)
            loss = criterion(output, train_target)
            loss.backward()
            optimizer.step()
            
            if verbose in (1,2):
                acc_loss = acc_loss + loss.item()
            
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
             
            if verbose == 2:
                if (e % 5) == 0:
                    print('epoch', e + 1, ':', acc_loss)
                    
    if verbose in (1, 2):
        if i in (0, 1, 2):
            fig, axs = plt.subplots(2, 1, figsize=(12,8))
            axs[0].plot(list(range(nb_epochs)), train_loss_list, label='Train loss')
            axs[0].plot(list(range(nb_epochs)), test_loss_list, label='Test loss')
            axs[0].legend()
            axs[1].plot(list(range(nb_epochs)), train_accu_list, label='Train accuracy')
            axs[1].plot(list(range(nb_epochs)), test_accu_list, label='Test accuracy')
            axs[1].legend()
            #plt.suptitle(end_date.date().strftime(format='%b %Y'))
            plt.show()

            
def test(model, X_test, y_test, threshold=None):
    
    pred = np.zeros((X_test.size(0), y_test.size(1)))
    prob = []
    
    for k in range(0, X_test.size(0)):
        output = model(X_test.narrow(0, k, 1))
        
        if threshold is None:
            _, pred_index = output.max(1)
            pred[k, pred_index.item()] = 1
            
        else:
            if k == 0:
                _, pred_index = output.max(1)
                pred[k, pred_index.item()] = 1
            else:
                out, pred_index = output.max(1)
                if out > threshold:
                    pred[k, pred_index.item()] = 1
                else:
                    pred[k] = pred[k-1]
                    
        prob.append(output.detach().numpy())
    return np.array(prob), pred

    
    
