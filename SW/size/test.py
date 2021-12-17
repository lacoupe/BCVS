from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import torch
from models import MLP, ConvNet, LSTM, SiameseLSTM
from train_test import train
from data import get_data, get_processed_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



def strat(features, target_prices, model_name='MLP', nb_epochs=50, 
            input_period=8, training_window=5, batch_size=1, verbose=0, eta=1e-3, dropout=0.1,
            weight_decay=1e-5, input_period_reg=10):

    num_features = len(features.columns)

    X, X_reg, y, _ = get_processed_data(features, target_prices, input_period, input_period_reg, training_window)

    mean = X.mean(dim=[0, 1], keepdim=True)
    std = X.std(dim=[0, 1], keepdim=True)
    X = X.sub_(mean).div_(std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

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

    train(model, X, y, nb_epochs=nb_epochs, device=device, batch_size=batch_size, eta=eta, verbose=verbose, weight_decay=weight_decay)

    # Today output
    df_input_period = features.iloc[-input_period:]

    X = df_input_period.values.reshape(input_period, num_features)
    X = torch.from_numpy(X).float()
    X = X.view(1, X.size(0), X.size(1)).to(device)
    X = X.sub_(mean).div_(std)

    model.eval()
    out = model(X)

    return pd.DataFrame(index=[model_name], columns=target_prices.columns, data=out.cpu().detach().numpy())

def run():
    _, target_prices, _, features = get_data()

    models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    # models_list = ['MLP']

    output = pd.DataFrame(index=['Ensemble'], columns=target_prices.columns)

    batch_size = 10
    training_window = 10
    nb_epochs = 10
    verbose = 3
    input_period = 20
    
    dropout = 0.2
    eta = 1e-3
    weight_decay = 1e-4

    for i, model_name in enumerate(models_list):
        print(model_name)
        df_output = strat(features=features, target_prices=target_prices, 
                          model_name=model_name, nb_epochs=nb_epochs, 
                          input_period=input_period, batch_size=batch_size,
                          training_window=training_window, verbose=verbose, 
                          eta=eta, weight_decay=weight_decay, dropout=dropout
                          )
        output = pd.concat([output, df_output])
        
        if i == 0:
            output.loc['Ensemble'] = output.loc[model_name]
        else:
            output.loc['Ensemble'] += output.loc[model_name]

    output.loc['Ensemble'] /= len(models_list)

    # Results
    print(32 * '*')
    print(output.round(4))
    print(32 * '*')

if __name__ == "__main__":
    run()