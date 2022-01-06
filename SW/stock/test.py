import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from helpers import last_month
from models import MLP, ConvNet, LSTM
from train_test import train
from data import get_price_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
import pdb

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__)))

def strat(df_input_all, best_pred, model_name='MLP', 
          nb_epochs=50, input_period=8, training_window=5, batch_size=1, verbose=0, eta=1e-3):

    last_date = df_input_all.index[-1]

    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    start_date = last_month(last_date - relativedelta(years=training_window))

    start_date_input = (start_date - relativedelta(months=input_period))

    df_output = best_pred.loc[start_date:].dropna()
    df_input = df_input_all.loc[start_date_input:df_output.index[-1]]
    X = []
    for idx in df_output.index:
        df_input_period = df_input.loc[:idx].asfreq('M', method='ffill').iloc[-input_period:]
        X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
        X.append(X_period)
    X = np.array(X)
    y = df_output.values

    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    train_mean = X.mean(dim=[0, 1, 2], keepdim=True)
    train_std = X.std(dim=[0, 1, 2], keepdim=True)
    X = X.sub_(train_mean).div_(train_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)

    # Initialize ML models only at first iteration
    dim1, dim2, dim3 = X.size(1), X.size(2), X.size(3)
    if model_name == 'MLP':
        model = MLP(dim1, dim2, dim3)
    elif model_name == 'ConvNet':
        model = ConvNet(dim1, dim2, dim3)
    elif model_name == 'LSTM':
        model = LSTM(input_size=num_tickers * num_features, output_size=num_tickers, device=device)
    model.to(device)

    train(model, X, y, nb_epochs, batch_size=batch_size, eta=eta, verbose=verbose)

    # Today output
    df_input_period = df_input_all.iloc[-input_period:]
    
    X = df_input_period.values.reshape(input_period, num_tickers, num_features)
    X = torch.from_numpy(X).float()
    X = X.view(1, X.size(0), X.size(1), X.size(2)).to(device)
    test_mean = X.mean(dim=[0, 1, 2], keepdim=True)
    test_std = X.std(dim=[0, 1, 2], keepdim=True)
    X = X.sub_(test_mean).div_(test_std)

    model.eval()
    out = model(X)

    return pd.DataFrame(index=[model_name], columns=best_pred.columns, data=out.cpu().detach().numpy())

def run():

    indice_weight, daily_returns, best_pred, df_X = get_price_data()
    today_tickers = list(indice_weight.iloc[-1].dropna().index)
    indice_weight = indice_weight[today_tickers]
    daily_returns = daily_returns[today_tickers]
    best_pred = best_pred[today_tickers]
    new_input_cols = df_X.columns.reindex(indice_weight.columns, level=0)
    df_X = df_X.reindex(columns=new_input_cols[0])

    models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    output = pd.DataFrame(index=['Ensemble'], columns=daily_returns.columns)
    
    batch_size = 10
    training_window = 5
    nb_epochs = 100
    verbose = 3
    input_period = 8
    eta = 1e-3

    for i, model_name in enumerate(models_list):
        print(model_name)
        df_output = strat(df_input_all=df_X, best_pred=best_pred,
                          model_name=model_name, nb_epochs=nb_epochs, 
                          input_period=input_period, batch_size=batch_size,
                          training_window=training_window, verbose=verbose, eta=eta)
        output = pd.concat([output, df_output])
        
        if i == 0:
            output.loc['Ensemble'] = output.loc[model_name]
        else:
            output.loc['Ensemble'] += output.loc[model_name]

    output.loc['Ensemble'] /= len(models_list)
    print(32 * '*')
    print(output.round(2))
    print(32 * '*')

if __name__ == "__main__":
    run()