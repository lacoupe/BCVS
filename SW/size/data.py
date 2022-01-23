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

def log_change(x):
    return np.log(x[-1] / x[0]) / len(x)


def pct_change(x):
    return (x[-1] - x[0]) / x[0]


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

    data = data.drop(columns=['EURCHF', 'USDCHF', 'SPI'])      

    technical_features = data[['SMALL_MID', 'LARGE']]

    mom5 = technical_features.rolling(5).apply(lambda x: log_change(x))
    mom5 = mom5.add_suffix(' mom5')

    mom21 = technical_features.rolling(21).apply(lambda x: log_change(x))
    mom21 = mom21.add_suffix(' mom21')

    mom63 = technical_features.rolling(63).apply(lambda x: log_change(x))
    mom63 = mom63.add_suffix(' mom63')

    vol21 = technical_features.pct_change().rolling(21).std()
    vol21 = vol21.add_suffix(' vol21')

    skew21 = technical_features.pct_change().rolling(21).skew()
    skew21 = skew21.add_suffix(' skew21')

    corr = target_prices.SMALL_MID.pct_change().rolling(21).corr(target_prices.LARGE.pct_change()).rename('correlation')

    features = pd.DataFrame(index=technical_features.index)
    for feature in ['MATERIALS', 'SP500', 'BRENT', 'FINANCIALS', 'GOLD', 'US 5YEAR', 
                                    'US 2YEAR', 'INDUSTRIALS', 'US 10YEAR', 'CONSUMER DIS.', 'HEALTH CARE', 
                                    'CONSUMER STAPLE', 'RUSSELL 2000', 'SILVER']:
        #features[feature] = fast_fracdiff(data[feature], 0.5)
        features[feature] = data[feature].rolling(5).apply(lambda x: log_change(x))

    features = pd.concat([mom5, mom21, mom63, vol21, skew21, corr, features, data[['SURPRISE', 'VSMI', 'VIX']]], axis=1).ewm(5).mean().dropna()

    features = features.drop(columns=['VIX', 'US 10YEAR', 'SP500', 'MATERIALS', 'CONSUMER DIS.', 'CONSUMER STAPLE', 'FINANCIALS'])

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