import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from data import get_data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

def backtest_strat(features, forward_weekly_returns,
                   training_window=4,
                   max_depth=None, min_samples_leaf=1, min_samples_split=2, n_estimators=100
                   ):

    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    df_prob_all = pd.DataFrame()
    all_end_dates = ['2013-07-01', '2014-01-01', '2014-07-01', '2015-01-01']
    for end_date in all_end_dates:

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date - relativedelta(years=training_window)

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = end_date - relativedelta(weeks=26)
        split_index = df_output.index.get_loc(start_date_test, method='bfill')
        index_test = df_output.iloc[split_index:].index

        threshold_return_diff = 0.0005
        features_std = pd.DataFrame(index=df_output.index, 
                                    data=PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(features.reindex(df_output.index)), 
                                    columns=features.columns)

        df_input = features_std.reindex(df_output.index)

        df_input_train = df_input.iloc[:split_index]
        df_output_train = df_output.iloc[:split_index]
        df_input_test = df_input.iloc[split_index:]

        forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
        index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
        X_train = df_input_train.values[index_train]
        X_test = df_input_test.values
        y_train = df_output_train.values[index_train]
        
        rf = RandomForestClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, 
                                                     min_samples_split=min_samples_split, n_estimators=n_estimators, random_state=1)
        rf.fit(X_train, y_train)

        # Get predictions
        prob = rf.predict_proba(X_test)[:, 1]
        df_prob = pd.DataFrame(index=index_test, data=prob)
        df_prob_all = pd.concat([df_prob_all, df_prob], axis=0)

    # print(len(df_prob_all))
    df_prob_all = df_prob_all[~df_prob_all.index.duplicated(keep='first')]
    # print(len(df_prob_all))
    scaler = MinMaxScaler()
    scaler.fit(df_prob_all.values.reshape(-1, 1))
    signal = scaler.transform(df_prob_all.values.reshape(-1, 1)).reshape(-1)
    # print(len(signal))
    return pd.DataFrame(index=df_prob_all.index, data=signal)

def tune_model():
    
    _, target_prices, features = get_data()
    target_prices = target_prices
    features = features

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)
    forward_weekly_returns = forward_weekly_returns.dropna()
    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    # Grid parameters
    rf_n_estimators = [50, 100, 500]
    rf_max_depth = [None, 5, 10, 20]
    rf_min_samples_leaf = [1, 5, 10]
    rf_min_samples_split = [2, 10, 20]

    training_window = 4
    
    tuning_list = []
    for n_estimators in tqdm(rf_n_estimators, position=0):
        for max_depth in tqdm(rf_max_depth, position=1, leave=False):
            for min_samples_leaf in tqdm(rf_min_samples_leaf, position=2, leave=False):
                for min_samples_split in tqdm(rf_min_samples_split, position=3, leave=False):
                    
                    df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns, max_depth=max_depth, training_window=training_window,
                                            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
                    init_length = len(df_prob)
                    df_prob = df_prob[(df_prob > 0.6) | (df_prob < 0.4)].dropna()
                    df_pred = (df_prob > 0.5).astype(int)
                    df_true = best_pred.reindex(df_pred.index)
                    tuning_list.append([n_estimators, max_depth, min_samples_leaf, min_samples_split, len(df_true) / init_length,
                                        100 * np.round(accuracy_score(df_pred.values, df_true.values), 4)
                                        ])
    print(df_prob.tail(20))
    print(pd.DataFrame(data=tuning_list, columns=['n_estimators', 'max_depth', 'min_samples_leaf', 'min_samples_split', 'Length', 'Accuracy']).sort_values('Accuracy').tail(20).to_string(index=False))

    # n_estimators=500, max_depth=20, min_leaf=1, min_split=20
if __name__ == "__main__":
    tune_model()