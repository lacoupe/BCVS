import numpy as np
import pandas as pd
import os

from helpers import clean_tickers
def get_price_data():

    file_path = os.path.join(os.path.dirname(__file__))
    indice_weight_path = file_path + '/data/Insurances.xlsx'
    indice_weight = pd.read_excel(indice_weight_path, sheet_name='raw_weight', skiprows=[1], index_col=0, parse_dates=True)
    indice_weight.drop(columns=indice_weight.columns[0], inplace=True)
    indice_weight.index.name = 'Date'
    indice_weight = indice_weight.T
    indice_weight.index = pd.to_datetime(indice_weight.index, format='%m/%d/%Y')
    indice_weight = indice_weight / 100
    indice_weight.replace(0, np.nan, inplace=True)
    clean_tickers(indice_weight)

    prices_path = file_path + '/data/Insurances.xlsx'
    prices = pd.read_excel(prices_path, sheet_name='performance', skiprows=[0,1,2,4,5,6], index_col=0)
    daily_returns = prices.pct_change().shift(1).fillna(0)
    clean_tickers(daily_returns)
    daily_returns = daily_returns[indice_weight.columns]

    monthly_returns = daily_returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
    max_ret = np.zeros_like(monthly_returns.values)
    max_ret[np.arange(len(max_ret)), monthly_returns.values.argmax(1)] = 1
    best_pred = pd.DataFrame(max_ret, columns=monthly_returns.columns, index=monthly_returns.index).astype(int).shift(-1)

    ratios_path = file_path + '/data/ratio_data.xlsx'
    all_ratios = pd.read_excel(ratios_path, sheet_name=None, skiprows=[0,1,2,4,5], parse_dates=True, index_col=0)
    for ratio in all_ratios:
        all_ratios[ratio] = all_ratios[ratio].shift(1)
    df_input = pd.concat(all_ratios, axis=1)
    df_input = df_input.swaplevel(0, 1, 1).sort_index(axis=1).fillna(method='ffill').fillna(0)
    clean_tickers(df_input, ratio=True)
    new_cols = df_input.columns.reindex(indice_weight.columns, level=0)
    df_input = df_input.reindex(columns=new_cols[0])

    tickers_name = ['BALOISE', 'GENERALI', 'HELVETIA', 'SCHWEIZERISCHE', 'SCOR', 'SWISS LIFE', 'SWISS REINSURANCE', 'VAUDOISE ASSURANCE', 'ZURICH INSURANCE', 'SWISS RE']
    indice_weight.columns = tickers_name
    daily_returns.columns = tickers_name
    best_pred.columns = tickers_name
    
    df_input.columns = df_input.columns.set_levels(tickers_name, level=0)


    return indice_weight, daily_returns, best_pred, df_input