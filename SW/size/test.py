from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import torch
from models import MLP, ConvNet, LSTM
from train_test import train
from data import get_data, get_processed_data
# from data import get_price_data, get_training_processed_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__)))


def strat(df_input_all, price, rebalance_freq, model_name='MLP', nb_epochs=50, 
            input_period=8, training_window=5, batch_size=1, verbose=0, eta_mlp=1e-3, eta_convnet=1e-3, eta_lstm=1e-3,
            weight_decay=1e-5, classification=True):

    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    X, y, _ = get_processed_data(df_input_all, price, rebalance_freq, input_period, training_window, classification)

    mean = X.mean(dim=[0, 1, 2], keepdim=True)
    std = X.std(dim=[0, 1, 2], keepdim=True)
    X = X.sub_(mean).div_(std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)

    if model_name == 'MLP':
        eta = eta_mlp
        model = MLP(dim1, dim2, dim3)
    elif model_name == 'ConvNet':
        eta = eta_convnet
        model = ConvNet(dim1, dim2, dim3)
    elif model_name == 'LSTM':
        eta = eta_lstm
        model = LSTM(input_size=num_tickers * num_features, output_size=num_tickers, device=device)

    model.to(device)

    train(model, X, y, nb_epochs=nb_epochs, device=device, batch_size=batch_size, eta=eta, verbose=verbose, weight_decay=weight_decay, classification=classification)

    # Today output
    if rebalance_freq == 'M':
        dayofweek = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        reb_freq_input = 'W-' + dayofweek[df_input_all.index[-1].weekday()][:3].upper()
        df_input_period = df_input_all.resample(reb_freq_input).mean().iloc[-input_period:]
    else:
        df_input_period = df_input_all.iloc[-input_period:]

    X = df_input_period.values.reshape(input_period, num_tickers, num_features)
    X = torch.from_numpy(X).float()
    X = X.view(1, X.size(0), X.size(1), X.size(2)).to(device)
    X = X.sub_(mean).div_(std)

    model.eval()
    out, _= model(X)

    return pd.DataFrame(index=[model_name], columns=price.columns, data=out.cpu().detach().numpy())

def run():
    price, _, df_X = get_data()

    # models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    models_list = ['MLP']

    output = pd.DataFrame(index=['Ensemble'], columns=price.columns)

    batch_size = 10
    training_window = 10
    nb_epochs = 2
    verbose = 3
    rebalance_freq = 'W-FRI'
    input_period_days = 42
    input_period_weeks = 8

    eta_mlp = 1e-3
    eta_convnet = 1e-3
    eta_lstm = 1e-3

    weight_decay = 1e-4

    if rebalance_freq == 'M':
        input_period = input_period_weeks
    else:
        input_period = input_period_days

    for i, model_name in enumerate(models_list):
        print(model_name)
        df_output = strat(df_input_all=df_X, price=price, rebalance_freq=rebalance_freq, 
                          model_name=model_name, nb_epochs=nb_epochs, 
                          input_period=input_period, batch_size=batch_size,
                          training_window=training_window, verbose=verbose, 
                          eta_mlp=eta_mlp, eta_convnet=eta_convnet, eta_lstm=eta_lstm, weight_decay=weight_decay,
                          )
        output = pd.concat([output, df_output])
        
        if i == 0:
            output.loc['Ensemble'] = output.loc[model_name]
        else:
            output.loc['Ensemble'] += output.loc[model_name]

    output.loc['Ensemble'] /= len(models_list)

    # Results
    print(32 * '*')
    print(output.round(6))
    print(32 * '*')

if __name__ == "__main__":
    run()