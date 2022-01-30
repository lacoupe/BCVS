import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from helpers import performance_plot, resume_backtest, annual_alpha_plot, price_to_perf, last_month, next_friday, prob_to_pred, printmetrics
from sklearn.metrics import classification_report
from models import MLP, RNN, GRU, LSTM
from train_test import train, test
from data import get_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, forward_weekly_returns, hidden_size=10, model_name='MLP',
                   nb_epochs=50, training_window=5, input_period=10, dropout=0.1,
                   batch_size=10, verbose=0, eta=1e-3, weight_decay=0, num_layers=2):

    print('Backtesting model ' + model_name)

    threshold_return_diff = 0.0005

    nbr_features = len(features.columns)

    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    # The moving window every 26 weeks
    df_prob_all = pd.DataFrame()
    all_end_dates = ['2015-07-01', '2016-01-01', '2016-07-01', '2017-01-01', 
                     '2017-07-01', '2018-01-01', '2018-07-01', '2019-01-01', 
                     '2019-07-01', '2020-01-01', '2020-07-01', '2021-01-01',
                     '2021-07-01']

    for end_date in tqdm(all_end_dates):

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date - relativedelta(years=training_window)
        start_date_input = start_date - relativedelta(days=input_period * 2)

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = end_date - relativedelta(months=6)
        split_index = df_output.index.get_loc(start_date_test, method='bfill')
        index_test = df_output.iloc[split_index:].index

        features_std = pd.DataFrame(index=best_pred.loc[start_date_input:end_date].index, 
                                    data=PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(features.reindex(best_pred.loc[start_date_input:end_date].index)), 
                                    columns=features.columns)

        if model_name == 'MLP':

            df_input = features_std.reindex(df_output.index)

            df_input_train = df_input.iloc[:split_index]
            df_output_train = df_output.iloc[:split_index]
            df_input_test = df_input.iloc[split_index:]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X_train = df_input_train.values[index_train]
            X_test = df_input_test.values

        else:

            df_input = features_std.loc[:end_date]

            df_output_train = df_output.iloc[:split_index]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X = []
            # print(index_train)
            for idx in df_output.index:
                # print(idx)
                # print(input_period)
                # print(df_input.head())
                # print(df_input.loc[:idx].head())
                # print(df_input.loc[:idx].iloc[-input_period:].head())
                df_input_period = df_input.loc[:idx].iloc[-input_period:]
                X_period = df_input_period.values.reshape(input_period, nbr_features)
                X.append(X_period)
            X = np.array(X)
            X_train = X[:split_index]
            X_train = X_train[index_train]
            X_test = X[split_index:]
        y_train = df_output_train.values[index_train]
        y_test = df_output_test.values

        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        if model_name == 'MLP':
            model = MLP(nbr_features, num_layers=num_layers, hidden_size=hidden_size, pdrop=dropout)
            
        elif model_name == 'RNN':
            model = RNN(nbr_features, hidden_size=5)

        elif model_name == 'GRU':
            model = GRU(nbr_features, hidden_size=9)

        # Train the model
        train(model, X_train, y_train, nb_epochs=nb_epochs, X_test=X_test, y_test=y_test, 
                batch_size=batch_size, eta=eta, weight_decay=weight_decay, verbose=verbose)
        # Get predictions
        prob = test(model, X_test)
        df_prob = pd.DataFrame(index=index_test, data=prob)
        df_prob_all = pd.concat([df_prob_all, df_prob], axis=0)

    df_prob_all = df_prob_all[~df_prob_all.index.duplicated(keep='first')]
    scaler = MinMaxScaler()
    scaler.fit(df_prob_all.values.reshape(-1, 1))
    signal = scaler.transform(df_prob_all.values.reshape(-1, 1)).reshape(-1)
    return pd.DataFrame(index=df_prob_all.index, data=signal)


def run_backtest():
    
    bench_price, target_prices, features = get_data()

    model_name = 'RNN'

    batch_size = 20
    verbose = 0
    training_window = 4

    nb_epochs = 50
    eta = 1e-3
    weight_decay = 1e-4
    dropout = 0.3
    
    hidden_size = 20
    num_layers = 1

    input_period = 10

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)

    seeds = list(range(5))
    for seed in seeds:

        torch.manual_seed(seed)
        print(f'SEED = {seed}')

        df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns,
                                model_name=model_name, nb_epochs=nb_epochs,
                                batch_size=batch_size, verbose=verbose, 
                                training_window=training_window, input_period=input_period, dropout=dropout,
                                eta=eta, weight_decay=weight_decay, hidden_size=hidden_size, num_layers=num_layers)

        # df_prob.iloc[:252].plot()
        # plt.show()

        portfolio = pd.DataFrame(index=df_prob.index, columns=['SMALL_MID', 'LARGE']).fillna(0)
        portfolio['SMALL_MID'] = (df_prob > 0.6).astype(int) - (df_prob < 0.4).astype(int)
        portfolio['LARGE'] = - portfolio.SMALL_MID

        class_type, class_count = np.unique(portfolio.values, axis=0, return_counts=True)
        weights = class_count / sum(class_count)
        print(class_type)
        print(weights)

        benchmark_portfolio = pd.read_excel('data/spiex_spi20_weights.xlsx', index_col=0)
        benchmark_portfolio = benchmark_portfolio.rename(columns={'SPIEX':'SMALL_MID', 'SPI20': 'LARGE'})
        benchmark_portfolio = benchmark_portfolio[['SMALL_MID', 'LARGE']]
        benchmark_portfolio = benchmark_portfolio.reindex(portfolio.index, method='bfill')

        portfolio = portfolio * 0.15 + benchmark_portfolio
        print(portfolio.index[0], portfolio.index[-1])
        df_resume = resume_backtest(portfolio, bench_price, target_prices)

        if seed == 0:
            df_final = df_resume
        else:
            df_final += df_resume
        print(df_resume)

    print(100 * '*')
    print((df_final / len(seeds)).round(2))

if __name__ == "__main__":
    run_backtest()