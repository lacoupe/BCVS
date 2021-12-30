import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from tqdm import tqdm
from helpers import performance_plot, resume_backtest, annual_alpha_plot, price_to_perf, last_month, next_friday, prob_to_pred, printmetrics
from models import MLP, LSTM# , ConvNet, LSTM
from train_test import train, test
from data import get_data
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score



import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def backtest_strat(features, forward_weekly_returns, model_name='MLP',
                   nb_epochs=50, training_window=5, input_period=10,
                   batch_size=10, verbose=0, eta=1e-3, weight_decay=0, dropout=0.1):


    nbr_features = len(features.columns)

    prob_output = []
    best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

    # The moving window every 26 weeks
    first_end_date = next_friday(features.index[input_period] + relativedelta(years=training_window))
    all_end_dates = best_pred.loc[first_end_date:].asfreq('W-FRI')[::26].index

    for i, end_date in enumerate(all_end_dates):

        start_date = next_friday(end_date - relativedelta(years=training_window))

        df_output = best_pred.loc[start_date:end_date]

        start_date_test = next_friday(end_date - relativedelta(weeks=26))
        split_index = df_output.index.get_loc(start_date_test) + 1

        threshold_return_diff = 0.0005

        if model_name == 'MLP':

            df_input = features.reindex(df_output.index)

            df_input_train = df_input.iloc[:split_index]
            df_output_train = df_output.iloc[:split_index]
            df_input_test = df_input.iloc[split_index:]
            df_output_test = df_output.iloc[split_index:]

            forward_weekly_returns_train = forward_weekly_returns.reindex(df_output_train.index)
            index_train = list(df_output_train.reset_index()[forward_weekly_returns_train.reset_index().abs_diff > threshold_return_diff].index)
            X_train = df_input_train.values[index_train]
            X_test = df_input_test.values

        elif model_name == 'LSTM':

            df_input = features.loc[:end_date]

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
        
        if i == 0:
            first_start_date_test = start_date_test
        
        # Transform Numpy arrays to Torch tensors
        X_train, y_train, X_test, y_test = torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
        
        # Standardize data within each feature 
        if model_name == 'MLP':
            std_dim = [0]
        else:
            std_dim = [0, 1]
        X_mean = X_train.mean(dim=std_dim, keepdim=True)
        X_std = X_train.std(dim=std_dim, keepdim=True)
        X_train = X_train.sub_(X_mean).div_(X_std)
        X_test = X_test.sub_(X_mean).div_(X_std)

        # Allocate the tensors to the GPU, if there is one
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        
        if model_name == 'MLP':
            model = MLP(nbr_features, pdrop=dropout)

        elif model_name == 'LSTM':
            model = LSTM(nbr_features, pdrop=dropout)

        model.to(device)

        # Train the model
        train(model, X_train, y_train, nb_epochs=nb_epochs, X_test=X_test, y_test=y_test, 
                batch_size=batch_size, eta=eta, weight_decay=weight_decay, verbose=verbose)
        # Get predictions
        prob = test(model, X_test)
        prob_output.append(prob)

    prob_output = np.array(prob_output).reshape(len(all_end_dates) * X_test.size(0))
    df_prob = pd.DataFrame(index=best_pred[first_start_date_test:end_date].iloc[1:].index, data=prob_output, columns=['prob'])

    return df_prob

def tune_model():

    print('GPU available :', torch.cuda.is_available())
    
    _, target_prices, features = get_data()
    target_prices = target_prices.loc['2015-01-01':]
    features = features.loc['2015-01-01':]

    model_name = 'LSTM'

    # Grid parameters
    learning_rates = [5e-4, 1e-3]
    weight_decays = [1e-3]
    dropouts = [0.1, 0.2]
    batch_sizes = [20, 40]
    nbs_epochs = [2, 3, 4]
    training_windows = [4]

    verbose = 0
    input_period = 21
    tuning_list = []
    for lr in tqdm(learning_rates):
        for w in weight_decays:
            for drop in dropouts:
                for b in batch_sizes:
                    for nb_epochs in nbs_epochs:
                        for training_window in training_windows:

                            forward_weekly_returns = target_prices.rolling(5).apply(lambda x: np.log(x[-1] / x[0]) / len(x)).shift(-5)
                            forward_weekly_returns['abs_diff'] = np.abs(forward_weekly_returns.SMALL_MID - forward_weekly_returns.LARGE)
                            best_pred = (forward_weekly_returns.SMALL_MID > forward_weekly_returns.LARGE).astype(int)

                            df_prob = backtest_strat(features=features, forward_weekly_returns=forward_weekly_returns,
                                                    model_name=model_name, nb_epochs=nb_epochs,
                                                    batch_size=b, verbose=verbose, 
                                                    training_window=training_window, input_period=input_period,
                                                    eta=lr, weight_decay=w, dropout=drop)

                            df_pred = (df_prob > 0.6).astype(int)
                            df_true = best_pred.reindex(df_pred.index)
                            tuning_list.append([lr, w, drop, nb_epochs, b, 
                                                np.round(accuracy_score(df_pred.values, df_true.values), 2)
                                            ])
    print(pd.DataFrame(data=tuning_list, columns=['Learning_rate', 'Weight_decay', 'Dropout', 
                                                    'Nb_epochs', 'Batch size','Accuracy']).sort_values('Accuracy').to_string(index=False))

if __name__ == "__main__":
    tune_model()