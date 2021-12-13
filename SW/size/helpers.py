import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
import os
from train_test import test
from sklearn.metrics import ConfusionMatrixDisplay


def plot_cm(model, X_test, X_test_reg, y_test, price):
    model.eval()
    output = test(model, X_test, X_test_reg)
    df_prob = pd.DataFrame(data=output.reshape(len(output), 2))
    df_pred = prob_to_pred_2(df_prob)

    ConfusionMatrixDisplay.from_predictions(y_test.cpu().detach().numpy().argmax(axis=1), 
                                            df_pred.values.argmax(axis=1), 
                                            display_labels=list(price.columns),
                                            cmap='Blues', colorbar=False
                                            )

    plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/confusion_matrix_' + model.__class__.__name__ + '.png'
    plt.title('Confusion Matrix ' + model.__class__.__name__ )
    plt.savefig(plot_path)
    plt.show()

def prob_to_pred(df_prob, threshold):
    df_pred = pd.DataFrame().reindex_like(df_prob).fillna(0)
    cols = df_pred.columns
    for k in range(0, len(df_pred)):
        if k == 0:
            pred_index = df_prob.iloc[k].argmax()
            df_pred.iloc[k][cols[pred_index]] = 1
        else:
            out = df_prob.iloc[k].max()
            pred_index = df_prob.iloc[k].argmax()
            if out > threshold:
                df_pred.iloc[k][cols[pred_index]] = 1
            else:
                df_pred.iloc[k] = df_pred.iloc[k-1]
    return df_pred


def prob_to_pred_2(df_prob):
    df_pred = pd.DataFrame().reindex_like(df_prob).fillna(0)
    cols = df_pred.columns
    for k in range(0, len(df_pred)):
        pred_index = df_prob.iloc[k].argmax()
        df_pred.iloc[k][cols[pred_index]] = 1
    return df_pred


def index_pred_plot(df_pred, daily_returns, first_date='2008-01-01', last_date='2010-01-01'):

    daily_ret = daily_returns.loc[first_date:last_date]
    daily_perf = np.log((daily_ret + 1).cumprod())
    df_pred_daily = df_pred.reindex(daily_ret.index, method='ffill').shift(1)
    daily_perf_pred = daily_perf.mul(df_pred_daily).replace(0., np.nan)
    indice_perf_pred = pd.Series(index=daily_perf_pred.index, dtype='float64', name='Prediction')
    for idx, row in daily_perf_pred[1:].iterrows():
        indice_perf_pred.loc[idx] = row[~row.isna()].item()

    _, ax = plt.subplots(figsize=(12,8))
    sns.lineplot(data=daily_perf.rolling(10).mean(), dashes=True, lw=1, zorder=1)
    ax.plot(indice_perf_pred.rolling(10).mean(), label='Prediction', c='r', lw=2, zorder=2)
    plt.legend()
    plt.show()

def last_month(date):
    year, month = date.year, date.month
    date_month = date.replace(day=calendar.monthrange(year, month)[1])
    return date_month

def last_friday(date):
    delta_days = 4 - date.weekday()
    if delta_days > 0:
        delta_days -= 7
    last_friday = date + relativedelta(days=delta_days)
    return last_friday

def next_friday(date):
    delta_days = 4 - date.weekday()
    if delta_days < 0:
        delta_days += 7
    last_friday = date + relativedelta(days=delta_days)
    return last_friday


def performance_plot(df_pred_dict, daily_returns, bench_price, log=True):
    tax = 0.0012
    perf_bench = price_to_perf(bench_price.loc[df_pred_dict['Ensemble'].index[0]:df_pred_dict['Ensemble'].index[-1]], log=log)
    plt.figure(figsize=(14,6))
    for model_name in df_pred_dict:
        perf_pred_net = pred_to_perf(df_pred_dict[model_name], daily_returns, tax, log=log)
        sns.lineplot(data=perf_pred_net.rolling(20).mean(), dashes=False, label=model_name)

    sns.lineplot(data=perf_bench.rolling(20).mean(), dashes=False, label='Benchmark')
    plt.legend(loc='best')
    plt.xlabel(None)
    plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/performance.png'
    plt.savefig(plot_path)
    plt.show()


def turnover(df_pred):
    return (df_pred.diff().fillna(0) != 0).any(axis=1).astype(int).sum()



def annual_alpha_plot(perf_bench, df_pred, daily_returns):
    
    tax = 0.0012
    perf_pred = pred_to_perf(df_pred, daily_returns,  tax)

    annual_returns_bench = perf_bench.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0])
    annual_returns_pred = perf_pred.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0])

    annual_diff = (annual_returns_pred - annual_returns_bench) * 100
    df_annual_diff = pd.DataFrame(columns=['year', 'alpha'])
    df_annual_diff['year'] = annual_diff.index.year
    df_annual_diff['year'] = df_annual_diff.year.apply(str)
    df_annual_diff['alpha'] = annual_diff.values
    df_annual_diff['sign'] = np.sign(df_annual_diff.alpha)

    _, ax = plt.subplots(figsize=(14,6))
    sns.barplot(data=df_annual_diff, x='year', y='alpha', hue='sign', dodge=False)
    ax.get_legend().remove()
    plt.ylabel('Alpha (%)')
    plt.xlabel('Year')
    plt.title('Annual Return of the model (net of tax) over Benchmark Return : ')
    plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/excess_return.png'
    plt.savefig(plot_path)
    plt.show()


def correlation(df_pred1, df_pred2):
    nb_match = 0
    L = len(df_pred1)
    for k in range(L):
        idx = np.where(df_pred1.iloc[k].values == 1)
        if df_pred2.iloc[k].values[idx] == 1.:
            nb_match += 1
    correlation = nb_match / L
    return correlation


def pred_to_perf(df_pred, daily_returns, tax=0., log=False):
    first_date = df_pred.index[0]
    last_date = df_pred.index[-1]
    daily_ret = daily_returns.loc[first_date:last_date]
    df_pred_daily = df_pred.reindex(daily_ret.index, method='ffill').shift(1)
    df_perf = (df_pred_daily * daily_ret).sum(axis=1)
    df_cost = (df_pred_daily.diff().fillna(0) != 0).any(axis=1).astype(int) * tax
    if log is False:
        perf = (1 + df_perf - df_cost).cumprod()
    else:
        perf = np.log(1 + df_perf - df_cost).cumsum()
    return perf


def pred_to_daily_ret(df_pred, daily_returns):
    first_date = df_pred.index[0]
    last_date = df_pred.index[-1]
    daily_ret = daily_returns[first_date:last_date]
    df_pred_daily = df_pred.reindex(daily_ret.index, method='ffill').shift(1)
    df_daily_ret = (df_pred_daily * daily_ret).sum(axis=1)
    return df_daily_ret


def price_to_perf(df, log=False):
    if log is False:
        perf = (1 + df.pct_change().fillna(0)).cumprod()
    else:
        perf = np.log(1 + df.pct_change().fillna(0)).cumsum()
    return perf


def perf_to_stat(perf_gross, perf_net):

    average_year_return_gross = perf_gross.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0]).mean() * 100
    average_year_return_net = perf_net.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0]).mean() * 100

    average_year_std = perf_gross.pct_change().std() * np.sqrt(256) * 100
    average_year_sharpe = average_year_return_net / average_year_std
    
    dd_window = 252
    roll_max = perf_gross.rolling(dd_window).max()
    daily_dd = perf_gross / roll_max - 1
    max_daily_dd = np.abs(daily_dd.rolling(dd_window, min_periods=1).min()).max() * 100

    return [average_year_return_gross, average_year_return_net, average_year_std, average_year_sharpe, max_daily_dd]


def perf_to_stat_2(df_pred, daily_returns, tax=0.0012):

    first_date = df_pred.index[0]
    last_date = df_pred.index[-1]
    daily_ret = daily_returns.loc[first_date:last_date]
    df_pred_daily = df_pred.reindex(daily_ret.index, method='ffill').shift(1)
    df_daily_perf= (df_pred_daily * daily_ret).sum(axis=1)
    df_cost = (df_pred_daily.diff().fillna(0) != 0).any(axis=1).astype(int) * tax
    perf_gross = (1 + df_daily_perf - df_cost).cumprod()
    
    average_year_return_gross = df_daily_perf.mean() * 252 * 100
    average_year_return_net = (df_daily_perf - df_cost).mean() * 252 * 100

    average_year_std = df_daily_perf.std() * np.sqrt(256) * 100
    average_year_sharpe = average_year_return_net / average_year_std
    
    dd_window = 252
    roll_max = perf_gross.rolling(dd_window).max()
    daily_dd = perf_gross / roll_max - 1
    max_daily_dd = np.abs(daily_dd.rolling(dd_window, min_periods=1).min()).max() * 100

    return [average_year_return_gross, average_year_return_net, average_year_std, average_year_sharpe, max_daily_dd]

def price_to_stats(price, df_pred=None):

    if df_pred is not None:
        first_date = df_pred.index[0]
        last_date = df_pred.index[-1]
        daily_ret = price.pct_change().shift(1).loc[first_date:last_date]
    else:
        daily_ret = price.pct_change().shift(1)
    perf = (daily_ret + 1).cumprod()

    average_year_return_gross = daily_ret.mean() * 252 * 100
    average_year_return_net = average_year_return_gross

    average_year_std = daily_ret.std() * np.sqrt(256) * 100
    average_year_sharpe = average_year_return_net / average_year_std

    dd_window = 252
    roll_max = perf.rolling(dd_window).max()
    daily_dd = perf / roll_max - 1
    max_daily_dd = np.abs(daily_dd.rolling(dd_window, min_periods=1).min()).max() * 100

    return [average_year_return_gross, average_year_return_net, average_year_std, average_year_sharpe, max_daily_dd]



def resume_backtest(df_pred_dict, bench_price, price):

    daily_returns = price.pct_change().shift(1)
    perf_bench = price_to_perf(bench_price.loc[next(iter(df_pred_dict.items()))[1].index[0]:next(iter(df_pred_dict.items()))[1].index[-1]], log=False)

    # bench_stats = perf_to_stat(perf_bench, perf_bench)
    bench_stats = price_to_stats(bench_price, df_pred_dict['Ensemble'])
    stats = []
    stats.append(bench_stats + [0])
    for model_name in df_pred_dict:
        # perf_gross = pred_to_perf(df_pred_dict[model_name], daily_returns, 0.)
        # perf_net = pred_to_perf(df_pred_dict[model_name], daily_returns, 0.0012)
        
        turnover_num = turnover(df_pred_dict[model_name])
        # stats.append(perf_to_stat(perf_gross, perf_net) + [turnover_num])
        stats.append(perf_to_stat_2(df_pred_dict[model_name], daily_returns, 0.0012) + [turnover_num])
    stats = np.array(stats)

    df_stats = pd.DataFrame(data=stats, columns=['Gross avg. annual return (%)', 'Net avg. annual return (%)', 
                                                'Avg. annual vol. (%)', 'Avg Sharpe ratio', 
                                                'Max. Drawdown (%)', 'Turnover'], 
                            index=['Benchmark SPI'] + list(df_pred_dict.keys())).round(2)

    df_stats.Turnover = df_stats.Turnover.apply(int)

    return df_stats



# def annual_alpha_plot_grossnet(perf_bench, df_pred, model, daily_returns):
#     perf_pred_gross = pred_to_perf(df_pred, daily_returns,  0.).rename('Model without transaction costs')
#     perf_pred_net = pred_to_perf(df_pred, daily_returns, 0.0012).rename('Model with transaction costs')

#     year_group_bench = perf_bench.resample('Y')
#     year_group_pred_gross = perf_pred_gross.resample('Y')
#     year_group_pred_net = perf_pred_net.resample('Y')

#     annual_returns_bench = (year_group_bench.last() - year_group_bench.first()) / year_group_bench.first()
#     annual_returns_pred_gross = (year_group_pred_gross.last() - year_group_pred_gross.first()) / year_group_pred_gross.first()
#     annual_returns_pred_net = (year_group_pred_net.last() - year_group_pred_net.first()) / year_group_pred_net.first()

#     annual_diff_gross = (annual_returns_pred_gross - annual_returns_bench) * 100
#     annual_diff_net = (annual_returns_pred_net - annual_returns_bench) * 100

#     df_annual_diff = pd.DataFrame(columns=['year', 'gross', 'net'])
#     df_annual_diff['year'] = annual_diff_net.index.year
#     df_annual_diff['year'] = df_annual_diff.year.apply(str)
#     df_annual_diff['gross'] = annual_diff_gross.values
#     df_annual_diff['net'] = annual_diff_net.values
#     df_annual_diff = df_annual_diff.melt(id_vars=['year'], var_name='gross_net', value_name='alpha')
#     df_annual_diff['sign'] = np.sign(df_annual_diff.alpha)

#     fig, ax = plt.subplots(figsize=(14,6))
#     sns.barplot(data=df_annual_diff, x='year', y='alpha', hue='gross_net', dodge=True)
#     ax.get_legend().set_title(None)
#     plt.ylabel('Alpha (%)')
#     plt.xlabel('Year')
#     plt.title('Annual Return of the model over Benchmark Return : ' + model)
#     plt.show()
