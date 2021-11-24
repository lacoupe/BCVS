import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
import calendar
from helpers import prob_to_perf, resume_backtest, last_month
from models import MLP, ConvNet, LSTM
from train_test import train, test
from data import get_price_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(df_input_all, best_pred, model_name='MLP', 
                   nb_epochs=50, nb_epochs_first=200, input_period=8, training_window=5, 
                   batch_size=1, verbose=0, eta=1e-3):

    print('Backtesting model ' + model_name)
    first_end_date = '2012-01-31'
    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    prob_output = []

    all_end_dates = best_pred.loc[first_end_date:].asfreq('6M').index

    for i, end_date in enumerate(tqdm(all_end_dates)):

        start_date = end_date - relativedelta(years=training_window)

        start_date_input = last_month(start_date - relativedelta(months=input_period))

        df_input = df_input_all.loc[start_date_input:end_date]
        df_output = best_pred.loc[start_date:end_date]

        X = []
        for idx in df_output.index:
            df_input_period = df_input.loc[:idx].iloc[-input_period:]
            X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
            X.append(X_period)

        X = np.array(X)
        y = df_output.values

        # Find the first prediction date
        if i == 0:
            first_start_date_test = end_date - relativedelta(months=5)

        start_date_test = (end_date - relativedelta(months=5))
        # Make sur the first test date is the end of the month
        year_test, month_test = start_date_test.year, start_date_test.month
        start_date_test = start_date_test.replace(day=calendar.monthrange(year_test, month_test)[1])
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
        if i == 0:
            if model_name == 'MLP':
                model = MLP(dim1, dim2, dim3)
            elif model_name == 'ConvNet':
                model = ConvNet(dim1, dim2, dim3)
            elif model_name == 'LSTM':
                model = LSTM(input_size=num_tickers * num_features, output_size=num_tickers, device=device)
        model.to(device)

        # More epochs needed for the first iteration 
        if i == 0:
            nb_epochs_all = nb_epochs_first
        else:
            nb_epochs_all = nb_epochs
        # Train the model
        train(model, X_train, y_train, nb_epochs_all, X_test, y_test, i, eta=eta, batch_size=batch_size, verbose=verbose)

        # Get predictions
        prob = test(model, X_test)
        prob_output.append(prob)

    prob_output = np.array(prob_output).reshape(len(all_end_dates) * X_test.size(0), y_test.size(1))
    df_prob = pd.DataFrame(index=best_pred[first_start_date_test:end_date].index, data=prob_output, columns=best_pred.columns)

    return df_prob

def run_backtest():

    print('GPU available :', torch.cuda.is_available())
    
    indice_weight, daily_returns, best_pred, df_X = get_price_data()

    models_list = ['MLP', 'ConvNet', 'LSTM']
    # models_list = ['MLP']
    
    df_prob_dict = {}

    batch_size = 10
    verbose = 0
    training_window = 4
    nb_epochs_first = 2
    nb_epochs = 1
    input_period = 8

    for i, model_name in enumerate(models_list):
        df_prob_dict[model_name] = backtest_strat(df_input_all=df_X, best_pred=best_pred, 
                                                  model_name=model_name, nb_epochs=nb_epochs, 
                                                  nb_epochs_first=nb_epochs_first, input_period=input_period, 
                                                  batch_size=batch_size, verbose=verbose, 
                                                  training_window=training_window)
        
        if i == 0:
            df_prob_dict['Ensemble'] = df_prob_dict[model_name].copy()
        else:
            df_prob_dict['Ensemble'] += df_prob_dict[model_name]
    
    
    # df_perf_dict = {}
    # for model_name in df_prob_dict:
    #     df_perf_dict[model_name] = prob_to_perf(df_prob_dict[model_name])

    daily_returns_backtest = daily_returns.loc[df_prob_dict['Ensemble'].index[0]:df_prob_dict['Ensemble'].index[-1]]
    bench_perf = (indice_weight.reindex(daily_returns_backtest.index, method='ffill').mul(daily_returns_backtest).sum(axis=1) + 1).cumprod()
    ensemble_perf = prob_to_perf(df_prob_dict['Ensemble'], indice_weight, daily_returns)
    # print('prob_dict', df_prob_dict['Ensemble'].head())
    # print('perf', ensemble_perf.head())
    print(145 * '*')
    print(resume_backtest(df_prob_dict, bench_perf, daily_returns, indice_weight))
    print(145 * '*')
    # perf = prob_to_perf
    # data_plot = pd.concat([perf, bench_perf], axis=1)
    # data_plot.columns = ['Strategie', 'Benchmark']

    # fig, ax = plt.subplots(figsize=(16,6))
    # sns.lineplot(data=data_plot.rolling(10).mean(), dashes=False)
    # plt.show()

if __name__ == "__main__":
    run_backtest()