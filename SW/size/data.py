import numpy as np
import pandas as pd
from helpers import RSI
import os, sys

def get_price_data():
    data_path = os.path.join(os.path.dirname(__file__)) + '/data/prices.csv'
    indices_price_excel = pd.read_csv(data_path, index_col=0, parse_dates=True)
    indices_price_excel.drop(columns=['SMIMC Index'], inplace=True)
    indices_price_excel.head()
    indices_price_excel.columns = ['SPI', 'MID', 'MID_SMALL', 'LARGE']

    bench_price = indices_price_excel['SPI']
    price = indices_price_excel[indices_price_excel.columns[1:]]

    mom12 = price.pct_change(periods=21 * 12)
    mom6 = price.pct_change(periods=21 * 6)
    mom1 = price.pct_change(periods=21 * 1)

    ma200 = np.log(price / price.rolling(window=200).mean())
    ma100 = np.log(price / price.rolling(window=100).mean())
    ma50 = np.log(price / price.rolling(window=50).mean())

    vol12 = price.rolling(window=21 * 12).std()
    vol6 = price.rolling(window=21 * 6).std()
    vol1 = price.rolling(window=21 * 1).std()

    ema_12 = price.ewm(span=10).mean()
    ema_26 = price.ewm(span=60).mean()
    MACD = ema_12 - ema_26

    RSI14 = RSI(price, 14)
    RSI9 = RSI(price, 9)
    RSI3 = RSI(price, 3)

    df_dict = {}
    df_input = pd.DataFrame()
    for col in price.columns:
        df_temp = pd.concat([ma50[col], ma100[col], ma200[col],
                            mom12[col], mom6[col], mom1[col],
                            vol12[col], vol6[col], vol1[col],
                            RSI14[col], RSI9[col], RSI3[col], 
                            MACD[col]], axis=1).loc['1997-01-01':].fillna(method='ffill')
        df_temp.columns = ['ma50', 'ma100', 'ma200', 'mom12', 'mom6', 'mom1', 
                        'vol12', 'vol6', 'vol1', 'RSI14', 'RSI9', 'MACD', 'RSI3']
        df_dict[col] = df_temp
        
    df_input = pd.concat(df_dict, axis=1).shift(1)

    return price, bench_price, df_input