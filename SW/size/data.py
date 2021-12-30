import numpy as np
import pandas as pd
import os
from helpers import last_friday, last_month
from dateutil.relativedelta import relativedelta
import torch
import pylab as pl


def RSI(price, window):
    price_diff = price.diff()
    gain = price_diff.mask(price_diff < 0, 0.0)
    loss = - price_diff.mask(price_diff > 0, -0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 1 - 1 / (1 + rs)
    return rsi.fillna(0)


# def get_price_data():
#     data_path = os.path.join(os.path.dirname(__file__)) + '/data/prices.csv'
#     indices_price_excel = pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
#     indices_price_excel.rename(columns={'SPI19 Index': 'SMALL', 'SMCI Index': 'MID', 'SPI21 Index': 'LARGE',
#                                         'MXEU0MT Index': 'Materials EU', 'MXEU0CS Index': 'Consumer Staple EU', 
#                                         'MXEU0IN Index': 'Industrials EU', 'MXEU0CD Index': 'Consumer Dis. EU',
#                                         'MXEU0HC Index': 'Health Care EU', 'MXEU0FN Index': 'Financials EU'}, inplace=True)
#     bench_price = indices_price_excel['SPI Index'].shift(1)

#     # target_prices = indices_price_excel[['SMALL', 'MID', 'LARGE']].shift(1)
#     target_prices = indices_price_excel[['SMALL', 'LARGE']].shift(1)
#     features = indices_price_excel[indices_price_excel.columns[1:-1]].shift(1)

#     weeky_returns = features.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))

#     ma50 = np.log(features / features.rolling(window=50).mean())

#     vol1 = features.rolling(window=21 * 1).std()

#     mom1 = np.log(features.rolling(21 * 1).apply(lambda x: x[-1] / x[0]))  / (21 * 1)
    
#     RSI5 = RSI(features, 5)

#     ema_12 = features.ewm(span=12).mean()
#     ema_26 = features.ewm(span=26).mean()
#     MACD = ema_12 - ema_26
#     MACD_diff = MACD - MACD.ewm(span=9).mean()

#     df_dict = {}
#     df_input = pd.DataFrame()
#     for col in features.columns:
#         df_temp = pd.concat([
#                             ma50[col], mom1[col], vol1[col], RSI5[col], MACD[col], MACD_diff[col], weeky_returns[col]
#                             ], axis=1)
#         df_temp.columns = ['ma50', 'mo1', 'vol1', 'RSI9', 'MACD', 'MACD_diff', 'weekly_returns']
#         df_dict[col] = df_temp

#     df_input = pd.concat(df_dict, axis=1).dropna(axis=0, how='any')

#     return features, bench_price, df_input, target_prices


def fast_fracdiff(x, d):

    T = len(x)
    np2 = int(2 ** np.ceil(np.log2(2 * T - 1)))
    k = np.arange(1, T)
    b = (1,) + tuple(np.cumprod((k - d - 1) / k))
    z = (0,) * (np2 - T)
    z1 = b + z
    z2 = tuple(x) + z
    dx = pl.ifft(pl.fft(z1) * pl.fft(z2))
    return np.real(dx[0:T])


def get_data():
    data_path = os.path.join(os.path.dirname(__file__)) + '/data/data.xlsx'

    data = pd.read_excel(data_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 8], sheet_name='features').fillna('ffill').shift(1).iloc[1:]

    data['MATERIALS EU'] *= data['EURCHF']
    data['CONSUMER STAPLE EU'] *= data['EURCHF']
    data['INDUSTRIALS EU'] *= data['EURCHF']
    data['CONSUMER DIS. EU'] *= data['EURCHF']
    data['HEALTH CARE EU'] *= data['EURCHF']
    data['FINANCIALS EU'] *= data['EURCHF']
    data['GOLD'] *= data['USDCHF']
    data['SILVER'] *= data['USDCHF']
    data['BRENT'] *= data['USDCHF']
    data['SP500'] *= data['USDCHF']
    data['RUSSELL 2000'] *= data['USDCHF']

    target_prices = pd.read_excel('data/data.xlsx', index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 8], sheet_name='target').fillna('ffill').shift(1).iloc[1:]

    bench_price = data['SPI']
    raw_features = data[data.columns.difference(['SPI', 'EURCHF', 'USDCHF'])]

    ma21 = np.log(raw_features[raw_features.columns[:-1]] / raw_features[raw_features.columns[:-1]].rolling(window=21).mean())
    ma21 = ma21.add_suffix(' ma21')

    ma50 = np.log(raw_features[raw_features.columns[:-1]] / raw_features[raw_features.columns[:-1]].rolling(window=50).mean())
    ma50 = ma50.add_suffix(' ma50')

    mom5 = raw_features[raw_features.columns[:-1]].rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom5 = mom5.add_suffix(' mom5')

    mom21 = raw_features[raw_features.columns[:-1]].rolling(21).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom21 = mom21.add_suffix(' mom21')

    ema_12 = raw_features[raw_features.columns[:-1]].ewm(span=12).mean()
    ema_26 = raw_features[raw_features.columns[:-1]].ewm(span=26).mean()
    MACD = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9).mean()
    MACD = MACD.add_suffix(' MACD')

    features = pd.concat([ma21, ma50, mom5, mom21, MACD, raw_features['SURPRISE']], axis=1).ewm(5).mean().dropna()

    # features_stationary = pd.DataFrame().reindex_like(features)
    # d_list = [0.5, 0.5, 0.4, 0.2, 0.2, 0.3, 0.3, 0.3, 0.2, 0., 0., 0., 0.3, 0.1, 0.5, 0.3, 0.3, 0.3, 0.4, 0.4, 0.5, 0.4]

    # # d_list = np.ones(len(features.columns))


    # if len(d_list) != len(features.columns):
    #     print(f'Error: d_list should have length {len(features.columns)} and has length {len(d_list)}')

    # for feature, d in zip(features.columns, d_list):
    #     features_stationary[feature] = fast_fracdiff(features[feature], d)

    return bench_price, target_prices, raw_features, features


def get_processed_data(features, target_prices, input_period, training_window):

    last_date_train = last_friday(target_prices.index[-input_period])
    
    num_features = len(features.columns)

    # log_weekly_returns = target_prices[:last_date_train].resample('W-FRI').apply(lambda x: np.log(x[-1] / x[0]) / len(x))

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)

    best_pred = (forward_weekly_returns.LARGE > forward_weekly_returns.SMALL_MID).astype(int)
    # best_pred = forward_weekly_returns.rank(axis=1).replace({1: 0., 2: 1.})

    start_date = last_friday(last_date_train - relativedelta(years=training_window))

    df_output = best_pred.loc[start_date:].dropna()
    # df_output_reg = forward_weekly_returns.shift(-1).loc[start_date:].dropna()

    start_date_input = target_prices.loc[:start_date].iloc[-252:].index[0]
    
    df_input = features.loc[start_date_input:df_output.index[-1]]
    # df_input_reg = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).loc[start_date_input:df_output.index[-1]]

    X = []
    # X_reg = []
    for idx in df_output.index:

        df_input_period = df_input.loc[:idx].iloc[-input_period:]
        # df_input_period_reg = df_input_reg.loc[:idx].iloc[-input_period_reg:]
        X_period = df_input_period.values.reshape(input_period, num_features)
        # X_period_reg = df_input_period_reg.values
        X.append(X_period)
        # X_reg.append(X_period_reg)

    X = np.array(X)
    # X_reg = np.array(X_reg)
    y = df_output.values
    # y_reg = df_output_reg.values

    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    return X, y