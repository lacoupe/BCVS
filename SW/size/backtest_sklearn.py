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
from datetime import datetime
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, forward_weekly_returns, training_window, model_name):

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
        start_date = next_friday(end_date - relativedelta(years=training_window))

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = next_friday(end_date - relativedelta(months=6))
        split_index = df_output.index.get_loc(start_date_test, method='bfill')
        index_test = df_output.iloc[split_index:].index

        features_std = pd.DataFrame(index=df_output.index, 
                                    data=PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(features.reindex(df_output.index)), 
                                    columns=features.columns)


        df_input = features_std.reindex(df_output.index)

        df_input_train = df_input.iloc[:split_index]
        df_output_train = df_output.iloc[:split_index]
        df_input_test = df_input.iloc[split_index:]
        df_output_test = df_output.iloc[split_index:]

        forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
        index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
        X_train = df_input_train.values[index_train]
        X_test = df_input_test.values

        y_train = df_output_train.values[index_train]
        y_test = df_output_test.values

        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        if model_name == 'RF':
            model = RandomForestClassifier(n_estimators=500, max_depth=20, min_samples_leaf=1, 
                                           min_samples_split=20, random_state=1)
        elif model_name == 'logreg':
            model = LogisticRegression(penalty='l2', class_weight='balanced')
        

        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        df_prob = pd.DataFrame(index=index_test, data=prob)
        df_prob_all = pd.concat([df_prob_all, df_prob], axis=0)

    df_prob_all = df_prob_all[~df_prob_all.index.duplicated(keep='first')]
    scaler = MinMaxScaler()
    scaler.fit(df_prob_all.values.reshape(-1, 1))
    signal = scaler.transform(df_prob_all.values.reshape(-1, 1)).reshape(-1)
    return pd.DataFrame(index=df_prob_all.index, data=signal)


def run_backtest():
    
    bench_price, target_prices, features = get_data()

    model_name = 'RF'
    training_window = 4

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)


    df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns,
                            model_name=model_name, training_window=training_window)
                            
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

    df_resume = resume_backtest(portfolio, bench_price, target_prices)

    print(df_resume)

if __name__ == "__main__":
    run_backtest()