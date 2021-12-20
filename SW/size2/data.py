import numpy as np
import pandas as pd
import os
from helpers import last_friday
import torch
from dateutil.relativedelta import relativedelta


def RSI(price, window):
    price_diff = price.diff()
    gain = price_diff.mask(price_diff < 0, 0.0)
    loss = - price_diff.mask(price_diff > 0, -0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 1 - 1 / (1 + rs)
    return rsi.fillna(0)


def get_data():
    data_path = os.path.join(os.path.dirname(__file__)) + '/data/data.csv'
    data = pd.read_csv(data_path, index_col=0, parse_dates=True)

    target_prices = data[['SMALL', 'LARGE']].fillna('ffill').shift(1).dropna()
    bench_price = data['SPI'].fillna('ffill').shift(1).dropna()
    features = data[data.columns[1:]].fillna('ffill').shift(1).dropna()

    change = features.diff()
    change = change.add_suffix(' change')

    ma20 = np.log(features / features.rolling(window=20).mean())
    ma20 = ma20.add_suffix(' ma20')
    ma50 = np.log(features / features.rolling(window=50).mean())
    ma50 = ma50.add_suffix(' ma50')

    vol21 = features.rolling(window=21).std()
    vol21 = vol21.add_suffix(' vol21')

    mom5 = features.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom5 = mom5.add_suffix(' mom5')
    mom10 = features.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom10 = mom10.add_suffix(' mom10')
    mom21 = features.rolling(21).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom21 = mom21.add_suffix(' mom21')

    RSI5 = RSI(features, 5)
    RSI5 = RSI5.add_suffix(' RSI5')

    RSI10 = RSI(features, 10)
    RSI10 = RSI10.add_suffix(' RSI10')

    ema_12 = features.ewm(span=12).mean()
    ema_26 = features.ewm(span=26).mean()
    MACD = ema_12 - ema_26
    MACD = MACD.add_suffix(' MACD')

    features = pd.concat([change, ma20, ma50, vol21, mom5, mom10, mom21, RSI5, RSI10, MACD], axis=1).dropna()

    return bench_price, target_prices, features


def get_processed_data(features, target_prices, training_window):

    last_date_train = last_friday(target_prices.index[-5])

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    best_pred = forward_weekly_returns.rank(axis=1).replace({1: 0., 2: 1.})

    start_date = last_friday(last_date_train - relativedelta(years=training_window))
    df_output = best_pred.loc[start_date:].dropna()

    df_input = features.loc[start_date:df_output.index[-1]]

    X = df_input.values
    y = df_output.values

    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    return X, y