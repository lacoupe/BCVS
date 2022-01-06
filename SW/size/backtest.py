import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from helpers import performance_plot, resume_backtest, annual_alpha_plot, price_to_perf, last_month, next_friday, prob_to_pred, printmetrics
from sklearn.metrics import classification_report
from models import MLP, GRU, LSTM
from train_test import train, test
from data import get_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
import matplotlib.pyplot as plt



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, forward_weekly_returns, hidden_size=10, model_name='MLP',
                   nb_epochs=50, training_window=5, input_period=10,
                   batch_size=10, verbose=0, eta=1e-3, weight_decay=0):

    print('Backtesting model ' + model_name)

    nbr_features = len(features.columns)

    prob_output = []
    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    # The moving window every 26 weeks
    first_end_date = next_friday(features.index[input_period] + relativedelta(years=training_window))
    all_end_dates = best_pred.loc[first_end_date:].asfreq('W-FRI')[::26].index

    for i, end_date in enumerate(tqdm(all_end_dates)):

        start_date = next_friday(end_date - relativedelta(years=training_window))

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = next_friday(end_date - relativedelta(weeks=26))
        split_index = df_output.index.get_loc(start_date_test) + 1

        threshold_return_diff = 0.0005

        if model_name == 'MLP':

            df_input = features.reindex(df_output.index)

            df_input_train = df_input.iloc[:split_index]
            df_output_train = df_output.iloc[:split_index]
            df_input_test = df_input.iloc[split_index:]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X_train = df_input_train.values[index_train]
            X_test = df_input_test.values

        else:

            df_input = features.loc[:end_date]

            df_output_train = df_output.iloc[:split_index]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X = []
            for idx in df_output.index:
                df_input_period = df_input.loc[:idx].iloc[-input_period:]
                X_period = df_input_period.values.reshape(input_period, nbr_features)
                X.append(X_period)
            X = np.array(X)
            X_train = X[:split_index]
            X_train = X_train[index_train]
            X_test = X[split_index:]
        y_train = df_output_train.values[index_train]
        y_test = df_output_test.values
        
        if i == 0:
            first_start_date_test = start_date_test
        
        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        
        # Standardize data within each feature 
        if model_name == 'MLP':
            std_dim = [0]
        else:
            std_dim = [0, 1]

        X_mean = X_train.mean(dim=std_dim, keepdim=True)
        X_std = X_train.std(dim=std_dim, keepdim=True)
        X_train = X_train.sub_(X_mean).div_(X_std)
        X_test = X_test.sub_(X_mean).div_(X_std)

        # Allocate the tensors to the GPU, if there is one
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        
        if model_name == 'MLP':
            model = MLP(nbr_features)
            
        elif model_name == 'GRU':
            model = GRU(nbr_features, hidden_size)

        elif model_name == 'LSTM':
            model = LSTM(nbr_features, hidden_size)

        model.to(device)

        # Train the model
        train(model, X_train, y_train, nb_epochs=nb_epochs, X_test=X_test, y_test=y_test, 
                batch_size=batch_size, eta=eta, weight_decay=weight_decay, verbose=verbose)
        # Get predictions
        prob = test(model, X_test)
        prob_output.append(prob)

    prob_output = np.array(prob_output).reshape(len(all_end_dates) * X_test.size(0))
    df_prob = pd.DataFrame(index=best_pred[first_start_date_test:end_date].iloc[1:].index, data=prob_output, columns=['prob'])

    return df_prob


def run_backtest():
    
    print('GPU available :', torch.cuda.is_available())
    
    bench_price, target_prices, features = get_data()
    target_prices = target_prices.loc['2014-01-01':]
    features = features.loc['2014-01-01':]

    model_name = 'GRU'

    batch_size = 20
    verbose = 0
    training_window = 4
    nb_epochs = 20
    eta = 1e-3
    weight_decay = 1e-4
    input_period = 21
    hidden_size = 10

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)
    # best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    seeds = list(range(5))
    for seed in seeds:

        torch.manual_seed(seed)

        df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns,
                                model_name=model_name, nb_epochs=nb_epochs,
                                batch_size=batch_size, verbose=verbose, 
                                training_window=training_window, input_period=input_period,
                                eta=eta, weight_decay=weight_decay, hidden_size=hidden_size)

        # _, ax = plt.subplots()
        # df_prob.iloc[-63:].plot(ax=ax)
        # plt.show()
        # df_prob = df_prob[(df_prob > 0.7) | (df_prob < 0.2)]
        # df_pred = (df_prob > 0.5).astype(int).fillna(method='ffill')
        # df_true = best_pred.reindex(df_pred.index)
        # printmetrics(df_true.values, df_pred.values)

        # df_pred_monday = df_pred[df_pred.index.weekday == 0].dropna()
        # df_true_monday = df_true.reindex(df_pred_monday.index)
        # print('MONDAY')
        # printmetrics(df_true_monday.values, df_pred_monday.values)

        # df_pred_tuesday = df_pred[df_pred.index.weekday == 1].dropna()
        # df_true_tuesday = df_true.reindex(df_pred_tuesday.index)
        # print('TUESDAY')
        # printmetrics(df_true_tuesday.values, df_pred_tuesday.values)

        # df_pred_wednesday = df_pred[df_pred.index.weekday == 2].dropna()
        # df_true_wednesday = df_true.reindex(df_pred_wednesday.index)
        # print('WEDNESDAY')
        # printmetrics(df_true_wednesday.values, df_pred_wednesday.values)

        # df_pred_thursday = df_pred[df_pred.index.weekday == 3].dropna()
        # df_true_thursday = df_true.reindex(df_pred_thursday.index)
        # print('THURSDAY')
        # printmetrics(df_true_thursday.values, df_pred_thursday.values)

        # df_pred_friday = df_pred[df_pred.index.weekday == 4].dropna()
        # df_true_friday = df_true.reindex(df_pred_friday.index)
        # print('FRIDAY')
        # printmetrics(df_true_friday.values, df_pred_friday.values)

        # print(classification_report(df_true.values, df_pred.values, labels=[0, 1], target_names=['LARGE', 'SMALL_MID']))
        
        # df_pred_weekly = df_pred.resample('W-WED').last()
        # df_pred_weekly = df_pred
        # portfolio = pd.DataFrame(index=df_pred_weekly.index, columns=['SMALL_MID', 'LARGE'])
        # portfolio['SMALL_MID'] = (df_pred_weekly == 1).astype(int)
        # portfolio['LARGE'] = (df_pred_weekly == 0).astype(int)

        portfolio = pd.DataFrame(index=df_prob.index, columns=['SMALL_MID', 'LARGE']).fillna(0)
        portfolio['SMALL_MID'] = (df_prob > 0.7).astype(int) - (df_prob < 0.3).astype(int)
        portfolio['LARGE'] = - portfolio.SMALL_MID

        benchmark_portfolio = pd.read_excel('data/spiex_spi20_weights.xlsx', index_col=0)
        benchmark_portfolio = benchmark_portfolio.rename(columns={'SPIEX':'SMALL_MID', 'SPI20': 'LARGE'})
        benchmark_portfolio = benchmark_portfolio[['SMALL_MID', 'LARGE']]
        benchmark_portfolio = benchmark_portfolio.reindex(portfolio.index, method='ffill')

        # portfolio = portfolio.replace({0: -1}) * 0.15 + benchmark_portfolio
        portfolio = portfolio * 0.15 + benchmark_portfolio
        # print(portfolio.head())

        df_resume = resume_backtest(portfolio, bench_price, target_prices)

        if seed == 0:
            df_final = df_resume
        else:
            df_final += df_resume
        print(df_resume)
        # print(df_prob)

    print( 100 * '*')
    print((df_final / len(seeds)).round(2))

if __name__ == "__main__":
    run_backtest()