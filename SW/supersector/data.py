import numpy as np
import pandas as pd
import os
from helpers import last_friday
import torch
from dateutil.relativedelta import relativedelta


# def RSI(price, window):
#     price_diff = price.diff()
#     gain = price_diff.mask(price_diff < 0, 0.0)
#     loss = - price_diff.mask(price_diff > 0, -0.0)
#     avg_gain = gain.rolling(window=window).mean()
#     avg_loss = loss.rolling(window=window).mean()
#     rs = avg_gain / avg_loss
#     rsi = 1 - 1 / (1 + rs)
#     return rsi.fillna(0)


def get_data():
    
    data_path = os.path.join(os.path.dirname(__file__)) + '/data/data.xlsx'
    data = pd.read_excel(data_path, index_col=0, skiprows=[0, 1, 2, 3, 4, 5, 6, 8], sheet_name='features').fillna(method='ffill').shift(1).iloc[1:]
    data = data.astype(float)

    data['MATERIALS'] *= data['EURCHF']
    data['CONSUMER STAPLE'] *= data['EURCHF']
    data['INDUSTRIALS'] *= data['EURCHF']
    data['CONSUMER DIS.'] *= data['EURCHF']
    data['HEALTH CARE'] *= data['EURCHF']
    data['FINANCIALS'] *= data['EURCHF']

    data['GOLD'] *= data['USDCHF']
    data['SILVER'] *= data['USDCHF']
    data['BRENT'] *= data['USDCHF']
    data['SP500'] *= data['USDCHF']
    data['RUSSELL 2000'] *= data['USDCHF']

    target_prices = data[['SMALL_MID', 'LARGE']]

    bench_price = data['SPI']

    raw_features = data[data.columns.difference(['SPI', 'EURCHF', 'USDCHF'])]

    raw_features = raw_features.drop(columns=['US 5YEAR', 'RUSSELL 2000', 'INDUSTRIALS'])

    technical_features = raw_features[raw_features.columns.difference(['SURPRISE'])]

    mom5 = technical_features.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom5 = mom5.add_suffix(' mom5')

    mom21 = technical_features.rolling(21).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom21 = mom21.add_suffix(' mom21')

    mom63 = technical_features.rolling(63).apply(lambda x: np.log(x[-1] / x[0]) / len(x))
    mom63 = mom63.add_suffix(' mom63')

    ema_12 = technical_features.ewm(span=12).mean()
    ema_26 = technical_features.ewm(span=26).mean()
    MACD = (ema_12 - ema_26) - (ema_12 - ema_26).ewm(span=9).mean()
    MACD = MACD.add_suffix(' MACD')

    # raw_features['SURPRISE'] = fast_fracdiff(raw_features['SURPRISE'], 0.5)

    features = pd.concat([mom5, mom21, mom63, MACD, raw_features['SURPRISE']], axis=1).ewm(5).mean().dropna()

    features = features.drop(columns=['FINANCIALS mom5', 'GOLD mom5', 'HEALTH CARE mom5',  
                                      'MATERIALS mom5',  'SILVER mom5',  'SMALL_MID mom5',  
                                      'US 10YEAR mom5',  'CONSUMER STAPLE mom21',  'GOLD mom21',  
                                      'SMALL_MID mom21',  'US 10YEAR mom21',  'US 2YEAR mom21',  
                                      'CONSUMER STAPLE mom63',  'FINANCIALS mom63',  
                                      'HEALTH CARE mom63',  'LARGE mom63',  'SILVER mom63',  
                                      'SMALL_MID mom63',  'US 10YEAR mom63', 'US 2YEAR mom63',])

    features = features.drop(columns=[ 'CONSUMER STAPLE mom5', 'LARGE mom5', 'CONSUMER DIS. mom21',  'SILVER mom21',  
                                        'BRENT mom63', 'GOLD mom63',  'MATERIALS mom63',  'CONSUMER DIS. MACD', 'SILVER MACD'])

    return bench_price, target_prices, features


def get_processed_data(features, target_prices, training_window):

    last_date_train = last_friday(target_prices.index[-5])

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)

    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    start_date = last_friday(last_date_train - relativedelta(years=training_window))

    df_output = best_pred.loc[start_date:last_date_train]
    df_input = features.reindex(df_output.index)

    X = df_input.values
    y = df_output.values

    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).float()

    return X, y