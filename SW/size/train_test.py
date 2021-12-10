import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt 
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import os

np.set_printoptions(precision=3, suppress=True)


def output_to_accu(model, X, X_reg, y):
    model.eval()
    nb_errors = 0
    for b in range(0, X.size(0)):
        output, _ = model(X.narrow(0, b, 1), X_reg.narrow(0, b, 1))
        _, predicted_classes = output.max(1)
        for k in range(1):
            if predicted_classes[k] != y.max(1)[1][b]:
                nb_errors = nb_errors + 1
            # if y[b + k, predicted_classes[k]] != 1.:
            #     nb_errors = nb_errors + 1
    accuracy = 100 * (1 - nb_errors / X.size(0))
    return accuracy


def output_to_loss(model, X, y):
    model.eval()
    loss = 0
    criterion = nn.BCELoss()
    for b in range(0, X.size(0)):
        output, _ = model(X.narrow(0, b, 1))
        loss += criterion(output, y.narrow(0, b, 1))
    return (loss / X.size(0)).cpu()


def train(model, X_train, y_train, nb_epochs, device, X_test=None, y_test=None, i=None, eta=1e-3, weight_decay=0, batch_size=1, verbose=0, classification=True):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)
    if classification:
        criterion = nn.BCELoss(reduction='none')
    else:
        criterion = nn.MSELoss()
    
    model.train()
    
    if verbose in (1, 2):
        train_accu_list = []
        train_loss_list = []
        test_accu_list = []
        test_loss_list = []
    
    train_set = TensorDataset(X_train, y_train)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    if classification:
        class_count = np.unique(y_train.cpu(), axis=0, return_counts=True)[1]
        weights = torch.tensor(class_count / sum(class_count)).to(device)

    for e in (tqdm(range(nb_epochs)) if (verbose == 3) else range(nb_epochs)):
        acc_loss = 0
        model.train()
        for train_input, train_target in train_loader:
            optimizer.zero_grad()
            output, _ = model(train_input)
            loss = criterion(output, train_target)
            if classification:
                loss = (loss * weights).mean()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            if verbose in (2, 4):
                acc_loss = acc_loss + loss.item()

        if verbose in (2, 4):
            if (e % 5) == 0:
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
        if i in (0, 1, 2):
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


def train_siamese(model, X_train, X_train_reg, y_train, y_train_reg, nb_epochs, device, 
                    X_test=None, y_test=None, eta=1e-3, weight_decay=0, 
                    batch_size=1, verbose=0, gamma=0.4):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=eta, weight_decay=weight_decay)
    criterion = nn.BCELoss(reduction='none')
    aux_criterion = nn.MSELoss()
    model.train()
    
    if verbose in (1, 2):
        train_accu_list = []
        train_loss_list = []
        test_accu_list = []
        test_loss_list = []
    

    train_set = TensorDataset(X_train, X_train_reg, y_train, y_train_reg)    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)

    class_count = np.unique(y_train.cpu(), axis=0, return_counts=True)[1]
    weights = torch.tensor(class_count / sum(class_count)).to(device)

    for e in (tqdm(range(nb_epochs)) if (verbose == 3) else range(nb_epochs)):
        acc_loss = 0
        acc_aux_loss = 0
        model.train()
        for i, (train_input, train_input_ret, train_target, returns) in enumerate(train_loader):
            optimizer.zero_grad()
            output, auxiliary = model(train_input, train_input_ret)
            loss = criterion(output, train_target)
            loss = (loss * weights).mean()

            auxiliary_1, auxiliary_2, auxiliary_3 = auxiliary.unbind(1)
            # auxiliary_1, auxiliary_2 = auxiliary.unbind(1)

            target_ret_1, target_ret_2, target_ret_3 = returns.unbind(1)
            # target_ret_1, target_ret_2 = returns.unbind(1)
            aux_loss = aux_criterion(auxiliary_1, target_ret_1) * weights[0] + \
                       aux_criterion(auxiliary_2, target_ret_2) * weights[1] + \
                       aux_criterion(auxiliary_3, target_ret_3) * weights[2]

            combined_loss = loss + gamma * aux_loss

            combined_loss.backward()
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            if verbose in (2, 4):
                acc_aux_loss +=  aux_loss.item()
                acc_loss += loss.item()

        if verbose in (2, 4):
            if (e % 1) == 0:
                print('epoch', e + 1, 
                      'class loss :', np.round(acc_loss, 4), 
                      'reg loss :', np.round(acc_aux_loss, 4), 
                      'accuracy :', np.round(output_to_accu(model, X_train, X_train_reg, y_train), 2), '%')
                print('prediction of returns \n ', auxiliary[:2].cpu().detach().numpy())
                print('true returns \n', returns[:2].cpu().detach().numpy())
                print('output \n', output[:2].cpu().detach().numpy())
                print('true prediction \n', train_target[:2].cpu().detach().numpy())

        if verbose in (1, 2):
            model.eval()
            train_accu = output_to_accu(model, X_train, X_train_reg, y_train)
            train_accu_list.append(train_accu)
            test_accu = output_to_accu(model, X_test, X_train_reg, y_test)
            test_accu_list.append(test_accu)
            
            # train_loss = output_to_loss(model, X_train, y_train).detach().numpy()
            # train_loss_list.append(train_loss)
            # test_loss = output_to_loss(model, X_test, y_test).detach().numpy()
            # test_loss_list.append(test_loss)
                    
    if verbose in (1, 2):
        _, axs = plt.subplots(2, 1, figsize=(12,8))
        # axs[0].plot(list(range(nb_epochs)), train_loss_list, label='Train loss')
        # axs[0].plot(list(range(nb_epochs)), test_loss_list, label='Test loss')
        # axs[0].legend()
        axs[1].plot(list(range(nb_epochs)), train_accu_list, label='Train accuracy')
        axs[1].plot(list(range(nb_epochs)), test_accu_list, label='Test accuracy')
        axs[1].legend()
        plt.xlabel('Epoch')
        plt.suptitle('Learning Curve ' + model.__class__.__name__, fontsize=15)
        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/learning_curve_' + model.__class__.__name__ + '.png'
        plt.savefig(plot_path)
        plt.show()


def test(model, X_test, X_test_reg):
    
    prob = []
    model.eval()
    for k in range(0, X_test.size(0)):
        output, _ = model(X_test.narrow(0, k, 1), X_test_reg.narrow(0, k, 1))
        prob.append(output.cpu().detach().numpy())

    return np.array(prob)
    
    
