import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from helpers import last_friday, last_month, annual_alpha_plot
from models import MLP, ConvNet, LSTM
from train_test import train
from data import get_price_data
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# import os, sys
# sys.path.append(os.path.join(os.path.dirname(__file__)))

def strat(df_input_all, price, rebalance_freq, model_name='MLP', nb_epochs=50, input_period=8, training_window=5, batch_size=1, verbose=0, eta=1e-3):

    last_date = price.index[-1]

    if rebalance_freq == 'M':
        last_date_train = last_month(last_date - relativedelta(weeks=input_period))
    else:
        last_date_train = last_friday(last_date - relativedelta(days=input_period))
    
    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    # Target data
    returns = price[:last_date_train].pct_change().shift(1).resample(rebalance_freq, convention='end').agg(lambda x: (x + 1).prod() - 1)
    best_pred = returns.rank(axis=1).replace({1: 0., 2: 0., 3: 1.}).shift(-1)

    if rebalance_freq == 'M':
        start_date = last_month(last_date_train - relativedelta(years=training_window))
    else:
        start_date = last_friday(last_date_train - relativedelta(years=training_window))

    if rebalance_freq =='M':
        start_date_input = (start_date - relativedelta(weeks=input_period)).replace(day=1) 
    else:
        start_date_input = start_date - relativedelta(days=input_period)
        start_date_input = start_date_input - relativedelta(days=(start_date_input.weekday()))

    df_output = best_pred.loc[start_date:].dropna()
    df_input = df_input_all.loc[start_date_input:df_output.index[-1]]
    
    X = []
    for idx in df_output.index:
        # If we rebalance monthly, the input data will be weekly data
        if rebalance_freq == 'M':
            dayofweek = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
            reb_freq_input = 'W-' + dayofweek[df_input_all.index[-1].weekday()][:3].upper()
            df_input_period = df_input.loc[:idx].asfreq(reb_freq_input, method='ffill').iloc[-input_period:]
        # If we rebalance weekly, the input data will be daily data
        else:
            df_input_period = df_input.loc[:idx].iloc[-input_period:]
            
        X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
        X.append(X_period)
    X = np.array(X)
    y = df_output.values
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    train_mean = X.mean(dim=[0, 1, 2], keepdim=True)
    train_std = X.std(dim=[0, 1, 2], keepdim=True)
    X = X.sub_(train_mean).div_(train_std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("The device used is", device)
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
    print(X.device)
    # Today output
    if rebalance_freq == 'M':
        dayofweek = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
        reb_freq_input = 'W-' + dayofweek[df_input_all.index[-1].weekday()][:3].upper()
        df_input_period = df_input.loc[:idx].asfreq(reb_freq_input, method='ffill').iloc[-input_period:]
    else:
        df_input_period = df_input_all.iloc[-input_period:]
    
    X = df_input_period.values.reshape(input_period, num_tickers, num_features)
    X = torch.from_numpy(X).float()
    X = X.view(1, X.size(0), X.size(1), X.size(2)).to(device)

    model.eval()
    out = model(X)

    return pd.DataFrame(index=[model_name], columns=price.columns, data=out.cpu().detach().numpy())

def run():
    price, _, df_X = get_price_data()

    # models_list = ['MLP', 'ConvNet', 'LSTM']
    models_list = ['ConvNet']
    output = pd.DataFrame(index=['Ensemble'], columns=price.columns)

    batch_size = 10
    training_window = 5
    nb_epochs = 1000
    verbose = 4
    rebalance_freq = 'W-FRI'
    input_period_days = 15
    input_period_weeks = 8
    eta = 5e-3
    # input_period = 15

    if rebalance_freq == 'M':
        input_period = input_period_weeks
    else:
        input_period = input_period_days

    for i, model_name in enumerate(models_list):
        df_output = strat(df_input_all=df_X, price=price, rebalance_freq=rebalance_freq, 
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