import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt 

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


def train(model, X_train, y_train, nb_epochs, i, eta=1e-3, mini_batch_size=1, verbose=0, X_test=None, y_test=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=eta)
    criterion = nn.BCELoss()
    model.train()
    
    if verbose in (1, 2):
        train_accu_list = []
        train_loss_list = []
        test_accu_list = []
        test_loss_list = []
     
    for e in range(nb_epochs):
        acc_loss = 0
        for b in range(0, X_train.size(0), mini_batch_size):
            optimizer.zero_grad()
            output = model(X_train.narrow(0, b, mini_batch_size))
            loss = criterion(output, y_train.narrow(0, b, mini_batch_size))
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
        if i in (0,1):
            fig, axs = plt.subplots(2, 1, figsize=(12,8))
            axs[0].plot(list(range(nb_epochs)), train_loss_list, label='Train loss')
            axs[0].plot(list(range(nb_epochs)), test_loss_list, label='Test loss')
            axs[0].legend()
            axs[1].plot(list(range(nb_epochs)), train_accu_list, label='Train accuracy')
            axs[1].plot(list(range(nb_epochs)), test_accu_list, label='Test accuracy')
            axs[1].legend()
            #plt.suptitle(end_date.date().strftime(format='%b %Y'))
            plt.show()

            
def test(model, X_test, y_test):
    pred = np.zeros((X_test.size(0), y_test.size(1)))
    prob = []
    for k in range(0, X_test.size(0)):
        output = model(X_test.narrow(0, k, 1))
        _, pred_index = output.max(1)
        pred[k, pred_index.item()] = 1
        prob.append(output.detach().numpy())
    return np.array(prob), pred

    
    
