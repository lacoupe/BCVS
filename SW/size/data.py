import numpy as np
import pandas as pd
import os
from helpers import last_friday, last_month
from dateutil.relativedelta import relativedelta
import torch


def RSI(price, window):
    price_diff = price.diff()
    gain = price_diff.mask(price_diff < 0, 0.0)
    loss = - price_diff.mask(price_diff > 0, -0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 1 - 1 / (1 + rs)
    return rsi.fillna(0)


def get_price_data():
    data_path = os.path.join(os.path.dirname(__file__)) + '/data/prices.csv'
    indices_price_excel = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
    indices_price_excel.head()
    indices_price_excel.rename(columns={'SPI19 Index': 'SMALL', 'SMCI Index': 'MID', 'SPI21 Index': 'LARGE',
                                        'MXEU0MT Index': 'Materials EU', 'MXEU0CS Index': 'Consumer Staple EU', 
                                        'MXEU0IN Index': 'Industrials EU', 'MXEU0CD Index': 'Consumer Dis. EU',
                                        'MXEU0HC Index': 'Health Care EU', 'MXEU0FN Index': 'Financials EU'}, inplace=True)
    bench_price = indices_price_excel['SPI Index'].shift(1)
    target_prices = indices_price_excel[['SMALL', 'MID', 'LARGE']].shift(1)
    features = indices_price_excel[indices_price_excel.columns[1:-1]].shift(1)

    weeky_returns = features.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))

    ma50 = np.log(features / features.rolling(window=50).mean())

    vol1 = features.rolling(window=21 * 1).std()

    mom1 = np.log(features.rolling(21 * 1).apply(lambda x: x[-1] / x[0]))  / (21 * 1)
    
    RSI5 = RSI(features, 5)

    ema_12 = features.ewm(span=12).mean()
    ema_26 = features.ewm(span=26).mean()
    MACD = ema_12 - ema_26
    MACD_diff = MACD - MACD.ewm(span=9).mean()

    df_dict = {}
    df_input = pd.DataFrame()
    for col in features.columns:
        df_temp = pd.concat([
                            ma50[col], mom1[col], vol1[col], RSI5[col], MACD[col], MACD_diff[col], weeky_returns[col]
                            ], axis=1)
        df_temp.columns = ['ma50', 'mo1', 'vol1', 'RSI9', 'MACD', 'MACD_diff', 'weekly_returns']
        df_dict[col] = df_temp

    df_input = pd.concat(df_dict, axis=1).dropna(axis=0, how='any')

    return features, bench_price, df_input, target_prices


def get_training_processed_data(df_input_all, target_prices, rebalance_freq, input_period, input_period_weeks, training_window, classification=True):

    last_date = target_prices.index[-1]

    if rebalance_freq == 'M':
        last_date_train = last_month(last_date - relativedelta(weeks=input_period))
    else:
        last_date_train = last_friday(target_prices.index[-input_period])
    
    num_tickers = len(df_input_all.columns.get_level_values(0).unique())
    num_features = len(df_input_all.columns.get_level_values(1).unique())

    # Target data
    # returns = price[:last_date_train].pct_change().shift(1).resample(rebalance_freq).agg(lambda x: (x + 1).prod() - 1)
    returns = target_prices[:last_date_train].resample(rebalance_freq).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    if classification:
        best_pred = returns.rank(axis=1).replace({1: 0., 2: 0., 3: 1.}).shift(-1)
        # best_pred = returns.rank(axis=1).replace({1: 0., 2: 1.}).shift(-1)
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
        start_date_input = target_prices.loc[:start_date].iloc[-input_period:].index[0]
    
    df_input = df_input_all.loc[start_date_input:df_output.index[-1]]
    df_input_reg = target_prices.resample('W-FRI').apply(lambda x: np.log(x[-1] / x[0]) / len(x)).loc[start_date_input:df_output.index[-1]]

    X = []
    X_reg = []
    for idx in df_output.index:
        # If we rebalance monthly, the input data will be weekly data
        if rebalance_freq == 'M':
            dayofweek = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
            reb_freq_input = 'W-' + dayofweek[df_input_all.index[-1].weekday()][:3].upper()
            df_input_period = df_input.loc[:idx].resample(reb_freq_input).mean().iloc[-input_period:]
        # If we rebalance weekly, the input data will be daily data
        else:
            df_input_period = df_input.loc[:idx].iloc[-input_period:]
            df_input_period_reg = df_input_reg.loc[:idx].iloc[-input_period_weeks:]
            
        X_period = df_input_period.values.reshape(input_period, num_tickers, num_features)
        X_period_reg = df_input_period_reg.values
        X.append(X_period)
        X_reg.append(X_period_reg)

    X = np.array(X)
    X_reg = np.array(X_reg)
    y = df_output.values
    y_reg = df_output_reg.values

    X, X_returns, y, y_reg = torch.from_numpy(X).float(), torch.from_numpy(X_reg).float(), torch.from_numpy(y).float(), torch.from_numpy(y_reg).float()

    return X, X_returns, y, y_reg