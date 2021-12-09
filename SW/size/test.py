from numpy.core.numeric import True_
import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from helpers import last_friday, last_month
from models import MLP, ConvNet, LSTM, MLP_REG, ConvNet_REG, LSTM_REG
from train_test import train
from data import get_price_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__)))

def get_training_processed_data(df_input_all, price, rebalance_freq, input_period, training_window, classification=True):

    last_date = price.index[-1]

    if rebalance_freq == 'M':
        last_date_train = last_month(last_date - relativedelta(weeks=input_period))
    else:
        last_date_train = last_friday(price.index[-input_period])
    
    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    # Target data
    # returns = price[:last_date_train].pct_change().shift(1).resample(rebalance_freq).agg(lambda x: (x + 1).prod() - 1)
    returns = price[:last_date_train].shift(1).resample(rebalance_freq).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    if classification:
        # best_pred = returns.rank(axis=1).replace({1: 0., 2: 0., 3: 1.}).shift(-1)
        best_pred = returns.rank(axis=1).replace({1: 0., 2: 1.}).shift(-1)
    else:
        best_pred = returns.shift(-1)

    if rebalance_freq == 'M':
        start_date = last_month(last_date_train - relativedelta(years=training_window))
    else:
        start_date = last_friday(last_date_train - relativedelta(years=training_window))

    df_output = best_pred.loc[start_date:].dropna()
    df_output_reg = returns.shift(-1).loc[start_date:].dropna()


    if rebalance_freq =='M':
        start_date_input = (start_date - relativedelta(weeks=input_period)).replace(day=1) 
    else:
        # start_date_input = start_date - relativedelta(days=input_period)
        # start_date_input = start_date_input - relativedelta(days=(start_date_input.weekday()))
        start_date_input = price.loc[:start_date].iloc[-input_period:].index[0]
    
    df_input = df_input_all.loc[start_date_input:df_output.index[-1]]

    X = []
    for idx in df_output.index:
        # If we rebalance monthly, the input data will be weekly data
        if rebalance_freq == 'M':
            dayofweek = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
            reb_freq_input = 'W-' + dayofweek[df_input_all.index[-1].weekday()][:3].upper()
            df_input_period = df_input.loc[:idx].resample(reb_freq_input).mean().iloc[-input_period:]
        # If we rebalance weekly, the input data will be daily data
        else:
            df_input_period = df_input.loc[:idx].iloc[-input_period:]
            
        X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
        X.append(X_period)

    X = np.array(X)
    y = df_output.values
    y_reg = df_output_reg.values

    X, y, y_reg = torch.from_numpy(X).float(), torch.from_numpy(y).float(), torch.from_numpy(y_reg).float()

    return X, y, y_reg


def strat(df_input_all, price, rebalance_freq, model_name='MLP', nb_epochs=50, 
            input_period=8, training_window=5, batch_size=1, verbose=0, eta_mlp=1e-3, eta_convnet=1e-3, eta_lstm=1e-3,
            weight_decay=1e-5, classification=True):

    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    X, y, _ = get_training_processed_data(df_input_all, price, rebalance_freq, input_period, training_window, classification)

    mean = X.mean(dim=[0, 1, 2], keepdim=True)
    std = X.std(dim=[0, 1, 2], keepdim=True)
    X = X.sub_(mean).div_(std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)
    if classification:
        if model_name == 'MLP':
            eta = eta_mlp
            model = MLP(dim1, dim2, dim3)
        elif model_name == 'ConvNet':
            eta = eta_convnet
            model = ConvNet(dim1, dim2, dim3)
        elif model_name == 'LSTM':
            eta = eta_lstm
            model = LSTM(input_size=num_tickers * num_features, output_size=num_tickers, device=device)
    else:
        if model_name == 'MLP':
            eta = eta_mlp
            model = MLP_REG(dim1, dim2, dim3)
        elif model_name == 'ConvNet':
            eta = eta_convnet
            model = ConvNet_REG(dim1, dim2, dim3)
        elif model_name == 'LSTM':
            eta = eta_lstm
            model = LSTM_REG(input_size=num_tickers * num_features, output_size=num_tickers, device=device)
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
    price, _, df_X = get_price_data()

    models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    # models_list = ['MLP']

    output = pd.DataFrame(index=['Ensemble'], columns=price.columns)

    batch_size = 10
    training_window = 10
    nb_epochs = 100
    verbose = 3
    rebalance_freq = 'W-FRI'
    input_period_days = 42
    input_period_weeks = 8

    eta_mlp = 1e-3
    eta_convnet = 1e-3
    eta_lstm = 1e-3

    weight_decay = 1e-4
    classification = True

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
                          classification=classification)
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