from matplotlib.pyplot import subplot_mosaic
from numpy.lib.shape_base import split
import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from helpers import performance_plot, resume_backtest, annual_alpha_plot, price_to_perf, last_month, next_friday, prob_to_pred, printmetrics
from models import MLP, RNN, GRU, LSTM
from train_test import train, test
from data import get_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PowerTransformer
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, forward_weekly_returns, model_name='MLP',
                   nb_epochs=50, training_window=5, input_period=10,
                   batch_size=10, verbose=0, eta=1e-3, weight_decay=0, dropout=0.1, hidden_size=10, num_layers=2):

    nbr_features = len(features.columns)

    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    df_prob_all = pd.DataFrame()
    all_end_dates = ['2013-07-01', '2014-01-01', '2014-07-01', '2015-01-01']
    for end_date in all_end_dates:

        torch.manual_seed(1)

        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date - relativedelta(years=training_window)
        start_date_input = start_date - relativedelta(days=input_period * 2)

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = end_date - relativedelta(weeks=26)
        split_index = df_output.index.get_loc(start_date_test, method='bfill')
        index_test = df_output.iloc[split_index:].index

        threshold_return_diff = 0.0005
        features_std = pd.DataFrame(index=best_pred.loc[start_date_input:end_date].index, 
                                    data=PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(features.reindex(best_pred.loc[start_date_input:end_date].index)), 
                                    columns=features.columns)
        if model_name == 'MLP':

            df_input = features_std.reindex(df_output.index)

            df_input_train = df_input.iloc[:split_index]
            df_output_train = df_output.iloc[:split_index]
            df_input_test = df_input.iloc[split_index:]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X_train = df_input_train.values[index_train]
            X_test = df_input_test.values

        else:

            df_input = features_std.loc[:end_date]

            df_output_train = df_output.iloc[:split_index]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X = []
            for idx in df_output.index:
                df_input_period = df_input.loc[:idx].iloc[-input_period:]
                X_period = df_input_period.values.reshape(input_period, nbr_features)
                X.append(X_period)
            X = np.array(X)
            X_train = X[:split_index]
            X_train = X_train[index_train]
            X_test = X[split_index:]

        y_train = df_output_train.values[index_train]
        y_test = df_output_test.values
        
        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()

        # Allocate the tensors to the GPU, if there is one
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        
        if model_name == 'MLP':
            model = MLP(nbr_features, num_layers=num_layers, hidden_size=hidden_size, pdrop=dropout)

        elif model_name == 'RNN':
            model = RNN(nbr_features, hidden_size)

        elif model_name == 'GRU':
            model = GRU(nbr_features, hidden_size)

        elif model_name == 'LSTM':
            model = LSTM(nbr_features, hidden_size)

        model.to(device)

        # Train the model
        train(model, X_train, y_train, nb_epochs=nb_epochs, X_test=X_test, y_test=y_test, 
                batch_size=batch_size, eta=eta, weight_decay=weight_decay, verbose=verbose)

        # Get predictions
        prob = test(model, X_test)
        df_prob = pd.DataFrame(index=index_test, data=prob)
        df_prob_all = pd.concat([df_prob_all, df_prob], axis=0)

    df_prob_all = df_prob_all[~df_prob_all.index.duplicated(keep='first')]
    scaler = MinMaxScaler()
    scaler.fit(df_prob_all.values.reshape(-1, 1))
    signal = scaler.transform(df_prob_all.values.reshape(-1, 1)).reshape(-1)
    return pd.DataFrame(index=df_prob_all.index, data=signal)

def tune_model():
    
    _, target_prices, features = get_data()
    target_prices = target_prices
    features = features

    forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
    forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)
    forward_weekly_returns = forward_weekly_returns.dropna()
    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    model_name = 'GRU'

    # fixed parameters
    nb_epochs = 30
    batch_size = 20
    weight_decay = 1e-4

    # Grid parameters
    num_layers_list = [1]
    hidden_size_list = [5, 6, 7, 8, 9, 10]
    eta_list = [1e-3]
    dropout_list = [0.1, 0.2, 0.3]
    dropout = 0.
    nb_epochs_list = [50]

    training_window = 4
    verbose = 0
    # input_period = 21
    input_period_list = [21]
    print(model_name)
    tuning_list = []
    for num_layers in tqdm(num_layers_list, position=0):
        for hidden_size in tqdm(hidden_size_list, position=1, leave=False):
            for eta in tqdm(eta_list, position=2, leave=False):
                for nb_epochs in tqdm(nb_epochs_list, position=3, leave=False):
                    for input_period in tqdm(input_period_list, position=4, leave=False):

                        df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns,
                                                model_name=model_name, nb_epochs=nb_epochs,
                                                batch_size=batch_size, verbose=verbose, 
                                                training_window=training_window, input_period=input_period, num_layers=num_layers,
                                                eta=eta, weight_decay=weight_decay, hidden_size=hidden_size, dropout=dropout)
                        init_length = len(df_prob)
                        df_prob = df_prob[(df_prob > 0.6) | (df_prob < 0.4)].dropna()
                        df_pred = (df_prob > 0.5).astype(int)
                        df_true = best_pred.reindex(df_pred.index)
                        tuning_list.append([num_layers, hidden_size, input_period, nb_epochs, len(df_true) / init_length,
                                            100 * np.round(accuracy_score(df_pred.values, df_true.values), 4)
                                            ])

    print(pd.DataFrame(data=tuning_list, columns=['num_layers', 'hidden_size', 'input_period', 'nb_epochs', 'Length', 'Accuracy']).sort_values('Accuracy').tail(20).to_string(index=False))

    # MLP: num_layers = 1, hidden_size = 20, dropout = 0.3, eta=0.001
    # RNN: hidden_size=10, input_period=42, epoch=60
if __name__ == "__main__":
    tune_model()