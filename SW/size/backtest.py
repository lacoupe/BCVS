import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import calendar
from helpers import performance_plot, resume_backtest, annual_alpha_plot, price_to_perf, last_month, next_friday, prob_to_pred
from models import MLP, ConvNet, LSTM
from train_test import train, test
from data import get_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, target_prices, model_name='MLP',
                   nb_epochs=50, input_period=8, training_window=5, 
                   batch_size=10, verbose=0, eta=1e-3, weight_decay=0):

    print('Backtesting model ' + model_name)

    num_features = len(features.columns)  

    # Target data
    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    best_pred = forward_weekly_returns.rank(axis=1).replace({1: 0., 2: 1.})
    
    prob_output = []
    
    # The moving window every 26 weeks
    first_end_date = next_friday(features.index[input_period] + relativedelta(years=training_window))
    all_end_dates = best_pred.loc[first_end_date:].asfreq('W-FRI')[::26].index

    for i, end_date in enumerate(tqdm(all_end_dates)):

        start_date = next_friday(end_date - relativedelta(years=training_window))
        # start_date_input = features.loc[:start_date].iloc[-input_period:].index[0]

        df_input = features.loc[:end_date]
        df_output = best_pred.loc[start_date:end_date]

        X = []
        for idx in df_output.index:
            df_input_period = df_input.loc[:idx].iloc[-input_period:]
            X_period = df_input_period.values.reshape(input_period, num_features)
            X.append(X_period)

        X = np.array(X)
        y = df_output.values
        
        start_date_test = next_friday(end_date - relativedelta(weeks=26))

        # Make sur the first test date is a friday
        # delta_days = 4 - start_date_test.weekday()
        # if delta_days < 0:
        #     delta_days += 7
        # start_date_test = start_date_test + relativedelta(days=delta_days)

        split_index = df_output.index.get_loc(start_date_test) + 1

        # print(start_date_input)
        # print(start_date)
        # print(start_date_test)
        # print(df_output[start_date_test : end_date])
        
        # print(start_date_test)
        # print(end_date)

        if i == 0:
            first_start_date_test = start_date_test
            test_index = df_output[start_date_test : end_date].iloc[1:].index
        test_index = test_index.union(df_output[start_date_test : end_date].iloc[1:].index)

        # Create train and test set
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]

        # print(len(df_output[start_date_test : end_date].iloc[1:]))
        # print(len(y_test))
        
        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        
        # Standardize data within each feature 
        train_mean = X_train.mean(dim=[0, 1, 2], keepdim=True)
        train_std = X_train.std(dim=[0, 1, 2], keepdim=True)
        X_train = X_train.sub_(train_mean).div_(train_std)
    
        test_mean = X_test.mean(dim=[0, 1, 2], keepdim=True)
        test_std = X_test.std(dim=[0, 1, 2], keepdim=True)
        X_test = X_test.sub_(test_mean).div_(test_std)

        # Allocate the tensors to the GPU, if there is one
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)

        dim1, dim2 = X_train.size(1), X_train.size(2)
        if model_name == 'MLP':
            model = MLP(dim1, dim2)

        elif model_name == 'ConvNet':
            model = ConvNet(dim1, dim2)

        elif model_name == 'LSTM':
            model = LSTM(input_size=num_features, output_size=2, device=device)

        model.to(device)

        # Train the model
        train(model, X_train, y_train, nb_epochs, X_test, y_test, i, eta=eta, weight_decay=weight_decay, batch_size=batch_size, verbose=verbose)

        # Get predictions
        prob = test(model, X_test)
        # print(len(prob))
        # print(len(y_test))
        prob_output.append(prob)

    # print(test_index)
    # print(best_pred[first_start_date_test:end_date].index)
    # print(all(test_index == best_pred[first_start_date_test:end_date].iloc[1:].index))

    prob_output = np.array(prob_output).reshape(len(all_end_dates) * X_test.size(0), y_test.size(1))
    df_prob = pd.DataFrame(index=best_pred[first_start_date_test:end_date].iloc[1:].index, data=prob_output, columns=best_pred.columns)

    return df_prob.resample('W-FRI').last()

def run_backtest():

    print('GPU available :', torch.cuda.is_available())
    
    bench_price, target_prices, _, features = get_data()

    # models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    models_list = ['ConvNet']
    df_pred_dict = {}
    df_prob_dict = {}

    threshold = 0.6
    batch_size = 5
    verbose = 0
    training_window = 5
    nb_epochs = 2
    input_period_days = 42
    input_period_reg = 8
    eta = 1e-3
    weight_decay = 1e-4

    input_period = input_period_days

    for i, model_name in enumerate(models_list):
        df_prob_dict[model_name] = backtest_strat(features=features, target_prices=target_prices,
                                                  model_name=model_name, nb_epochs=nb_epochs, 
                                                  input_period=input_period, 
                                                  batch_size=batch_size, verbose=verbose, 
                                                  training_window=training_window,
                                                  eta=eta, weight_decay=weight_decay)
        df_pred_dict[model_name] = prob_to_pred(df_prob_dict[model_name], threshold)
        
        if i == 0:
            df_prob_dict['Ensemble'] = df_prob_dict[model_name].copy()
        else:
            df_prob_dict['Ensemble'] += df_prob_dict[model_name]
            
    df_prob_dict['Ensemble'] /= len(models_list)
    df_pred_dict['Ensemble'] = prob_to_pred(df_prob_dict[model_name], threshold)

    df_resume = resume_backtest(df_pred_dict, bench_price, target_prices)
    print(df_resume)

    daily_returns = target_prices.pct_change().shift(1)
    perf_bench = price_to_perf(bench_price.loc[df_pred_dict['Ensemble'].index[0]:df_pred_dict['Ensemble'].index[-1]], log=False)
    performance_plot(df_pred_dict, daily_returns, bench_price, log=True)
    annual_alpha_plot(perf_bench, df_pred_dict['Ensemble'], daily_returns)

if __name__ == "__main__":
    run_backtest()