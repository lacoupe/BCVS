import numpy as np
import pandas as pd
import os


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
    indices_price_excel = pd.read_csv(data_path, index_col=0, parse_dates=True)
    # indices_price_excel.drop(columns=['SMIMC Index'], inplace=True)
    indices_price_excel.head()
    indices_price_excel.columns = ['SPI', 'SMALL', 'MID', 'LARGE']

    bench_price = indices_price_excel['SPI']
    # price = indices_price_excel[indices_price_excel.columns[1:]].shift(1)
    price = indices_price_excel[['SMALL', 'LARGE']].shift(1)

    weeky_returns = price.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x))

    ma200 = np.log(price / price.rolling(window=200).mean())
    ma100 = np.log(price / price.rolling(window=100).mean())
    ma50 = np.log(price / price.rolling(window=50).mean())

    # ma200 = price.rolling(window=200).mean()
    # ma100 = price.rolling(window=100).mean()
    # ma50 = price.rolling(window=50).mean()

    # vol12 = np.log(price.rolling(window=21 * 12).std() / price.rolling(window=21 * 12).std().shift(1))
    # vol6 = np.log(price.rolling(window=21 * 6).std() / price.rolling(window=21 * 6).std().shift(1))
    # vol1 = np.log(price.rolling(window=21 * 1).std() / price.rolling(window=21 * 1).std().shift(1))

    vol12 = price.rolling(window=21 * 12).std()
    vol6 = price.rolling(window=21 * 6).std()
    vol1 = price.rolling(window=21 * 1).std()

    # mom12 = price.pct_change(periods=21 * 12)
    # mom6 = price.pct_change(periods=21 * 6)
    # mom1 = price.pct_change(periods=21 * 1)

    mom12 = np.log(price.rolling(21 * 12).apply(lambda x: x[-1] / x[0]))  / (21 * 12)
    mom6 = np.log(price.rolling(21 * 6).apply(lambda x: x[-1] / x[0]))  / (21 * 6)
    mom1 = np.log(price.rolling(21 * 1).apply(lambda x: x[-1] / x[0]))  / (21 * 1)

    upper_bollinger = price.rolling(20).mean() + 2 * price.rolling(20).std()
    lower_bollinger = price.rolling(20).mean() - 2 * price.rolling(20).std()
    upper_boll_diff = np.log(price / upper_bollinger)
    lower_boll_diff = np.log(price / lower_bollinger)
    
    RSI14 = RSI(price, 14)
    RSI9 = RSI(price, 9)
    # RSI3 = RSI(price, 3)

    ema_12 = price.ewm(span=12).mean()
    ema_26 = price.ewm(span=26).mean()
    MACD = ema_12 - ema_26
    MACD_diff = MACD - MACD.ewm(span=9).mean()

    df_dict = {}
    df_input = pd.DataFrame()
    for col in price.columns:
        df_temp = pd.concat([
                            ma50[col], ma100[col], ma200[col],
                            mom12[col], mom6[col], mom1[col],
                            vol12[col], vol6[col], vol1[col],
                            RSI14[col], RSI9[col], #RSI3[col], 
                            MACD[col], MACD_diff[col], upper_boll_diff[col], lower_boll_diff[col], weeky_returns[col]
                            ], axis=1).iloc[252:]
        df_temp.columns = ['ma50', 'ma100', 'ma200', 'mom12', 'mom6', 'mom1',
                           'vol12', 'vol6', 'vol1', 'RSI14', 'RSI9', 'MACD', 'MACD_diff', 'upper_boll_diff', 'lower_boll_diff', 'weekly_returns'
                           ]
        # df_temp = price[col]
        # df_temp.columns = ['price']
        df_dict[col] = df_temp

    df_input = pd.concat(df_dict, axis=1).dropna(axis=0, how='any')

    return price, bench_price, df_input