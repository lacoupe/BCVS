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
from data import get_price_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(df_input_all, price, rebalance_freq, model_name='MLP', 
                   nb_epochs=50, nb_epochs_first=200, input_period=8, training_window=5, 
                   batch_size=1, verbose=0, threshold=0.4, eta=1e-3, weight_decay=0):

    print('Backtesting model ' + model_name)

    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())
    
    # Target data
    returns = price.pct_change().shift(1).resample(rebalance_freq).agg(lambda x: (x + 1).prod() - 1)
    best_pred = returns.rank(axis=1).replace({1: 0., 2: 0., 3: 1.}).shift(-1)
    
    prob_output = []
    
    if rebalance_freq == 'M':
        # The moving window every 6 month
        first_end_date = last_month(df_input_all.index[0] + relativedelta(years=training_window) + relativedelta(weeks=input_period))
        all_end_dates = best_pred.loc[first_end_date:].asfreq('6M').index
    else:
        # The moving window every 26 weeks
        first_end_date = next_friday(df_input_all.index[input_period] + relativedelta(years=training_window))#  + relativedelta(days=input_period))
        all_end_dates = best_pred.loc[first_end_date:].asfreq('W-FRI')[::26].index

    for i, end_date in enumerate(tqdm(all_end_dates)):

        # The first date input must before the first date output
        if rebalance_freq =='M':
            # Make sur the input period start the 1st of the month
            start_date = last_month(end_date - relativedelta(years=training_window))
            start_date_input = (start_date - relativedelta(weeks=input_period)).replace(day=1) 
        else:
            start_date = next_friday(end_date - relativedelta(years=training_window))
            # start_date_input = start_date - relativedelta(days=input_period)
            start_date_input = df_input_all.loc[:start_date].iloc[-input_period:].index[0]
            # print('start_date_input', start_date_input)
            # Make sur the input period start a monday
            # start_date_input = start_date_input - relativedelta(days=(start_date_input.weekday()))

        # print('end_date', end_date)
        # print('start_date', start_date)
        # print('start_Date_input', start_date_input)
        df_input = df_input_all.loc[start_date_input:end_date]
        df_output = best_pred.loc[start_date:end_date]
        # print(df_input.head())
        X = []
        for idx in df_output.index:
            # print('index', idx)
            # If we rebalance monthly, the input data will be weekly data
            if rebalance_freq == 'M':
                df_input_period = df_input.loc[:idx].resample('W').mean().iloc[-input_period:]
            # If we rebalance weekly, the input data will be daily data
            else:
                df_input_period = df_input.loc[:idx].iloc[-input_period:]
                # print(df_input_period.head())

            # print(len(df_input_period))
            X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
            X.append(X_period)

        X = np.array(X)
        y = df_output.values
        
        # Find the first prediction date
        if i == 0:
            if rebalance_freq == 'M':
                first_start_date_test = end_date - relativedelta(months=5)
            else:
                first_start_date_test = end_date - relativedelta(weeks=25)
        
        if rebalance_freq == 'M':
            start_date_test = (end_date - relativedelta(months=5))
            # Make sur the first test date is the end of the month
            year_test, month_test = start_date_test.year, start_date_test.month
            start_date_test = start_date_test.replace(day=calendar.monthrange(year_test, month_test)[1])
            split_index = df_output.index.get_loc(start_date_test)    
        else:
            start_date_test = end_date - relativedelta(weeks=25)
            # Make sur the first test date is a friday
            delta_days = 4 - start_date_test.weekday()
            if delta_days < 0:
                delta_days += 7
            start_date_test = start_date_test + relativedelta(days=delta_days)
            split_index = df_output.index.get_loc(start_date_test)  
        
        # Create train and test set
        X_train, y_train = X[:split_index], y[:split_index]
        X_test, y_test = X[split_index:], y[split_index:]
        
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

        # Initialize ML models only at first iteration
        dim1, dim2, dim3 = X_train.size(1), X_train.size(2), X_train.size(3)
        if model_name == 'MLP':
            model = MLP(dim1, dim2, dim3)
        elif model_name == 'ConvNet':
            model = ConvNet(dim1, dim2, dim3)
        elif model_name == 'LSTM':
            model = LSTM(input_size=num_tickers * num_features, output_size=num_tickers, device=device)
        model.to(device)

        # More epochs needed for the first iteration 

        # Train the model
        train(model, X_train, y_train, nb_epochs, X_test, y_test, i, eta=eta, weight_decay=weight_decay, batch_size=batch_size, verbose=verbose)

        # Get predictions
        prob = test(model, X_test, y_test, threshold=threshold)
        prob_output.append(prob)

    # pred_output = np.array(pred_output).reshape(len(all_end_dates) * X_test.size(0), y_test.size(1))
    # df_pred = pd.DataFrame(index=best_pred[first_start_date_test:end_date].index, data=pred_output, columns=best_pred.columns)
    prob_output = np.array(prob_output).reshape(len(all_end_dates) * X_test.size(0), y_test.size(1))
    df_prob = pd.DataFrame(index=best_pred[first_start_date_test:end_date].index, data=prob_output, columns=best_pred.columns)
    
    return df_prob

def run_backtest():

    print('GPU available :', torch.cuda.is_available())
    
    price, bench_price, df_X = get_price_data()

    # models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP', 'ConvNet']
    models_list = ['MLP']
    df_pred_dict = {}
    df_prob_dict = {}

    threshold = 0.8
    batch_size = 5
    verbose = 4
    training_window = 5
    nb_epochs = 1
    rebalance_freq = 'W-FRI'
    input_period_days = 42
    input_period_weeks = 8
    eta = 1e-3
    weight_decay = 1e-4

    if rebalance_freq == 'M':
        input_period = input_period_weeks
    else:
        input_period = input_period_days

    for i, model_name in enumerate(models_list):
        df_prob_dict[model_name] = backtest_strat(df_input_all=df_X, price=price, rebalance_freq=rebalance_freq, 
                                                  model_name=model_name, nb_epochs=nb_epochs, 
                                                  input_period=input_period, 
                                                  batch_size=batch_size, verbose=verbose, 
                                                  training_window=training_window, threshold=threshold, 
                                                  eta=eta, weight_decay=weight_decay)
        df_pred_dict[model_name] = prob_to_pred(df_prob_dict[model_name], threshold)
        
        if i == 0:
            df_prob_dict['Ensemble'] = df_prob_dict[model_name].copy()
        else:
            df_prob_dict['Ensemble'] += df_prob_dict[model_name]
            
    df_prob_dict['Ensemble'] /= len(models_list)
    df_pred_dict['Ensemble'] = prob_to_pred(df_prob_dict[model_name], threshold)

    df_resume = resume_backtest(df_pred_dict, bench_price, price)
    print(df_resume)

    daily_returns = price.pct_change().shift(1)
    perf_bench = price_to_perf(bench_price.loc[df_pred_dict['Ensemble'].index[0]:df_pred_dict['Ensemble'].index[-1]], log=False)
    performance_plot(df_pred_dict, daily_returns, bench_price, log=True)
    annual_alpha_plot(perf_bench, df_pred_dict['Ensemble'], daily_returns)
    # index_pred_plot(df_pred_dict['Ensemble'], daily_returns)

if __name__ == "__main__":
    run_backtest()