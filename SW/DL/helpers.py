import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dateutil.relativedelta import relativedelta
import calendar
import os


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

def RSI(price, window):
    price_diff = price.diff()
    gain = price_diff.mask(price_diff < 0, 0.0)
    loss = - price_diff.mask(price_diff > 0, -0.0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 1 - 1 / (1 + rs)
    return rsi

def perf_to_stat(perf_gross, perf_net):
    
    year_group_gross = perf_gross.resample('Y')
    year_group_net = perf_net.resample('Y')

    average_year_return_gross = ((year_group_gross.last() - year_group_gross.first()) / year_group_gross.first()).mean() * 100
    average_year_return_net = ((year_group_net.last() - year_group_net.first()) / year_group_net.first()).mean() * 100

    average_year_std = perf_gross.pct_change().std() * np.sqrt(256) * 100
    average_year_sharpe = average_year_return_net / average_year_std
    
    dd_window = 252
    roll_max = perf_gross.rolling(dd_window).max()
    daily_dd = perf_gross / roll_max - 1
    max_daily_dd = np.abs(daily_dd.rolling(dd_window, min_periods=1).min()).max() * 100

    return [average_year_return_gross, average_year_return_net, average_year_std, average_year_sharpe, max_daily_dd]


def performance_plot(df_pred, model, daily_returns, bench_price):
    perf_pred_gross = pred_to_perf(df_pred, daily_returns, 0., log=True).rename('Model without transaction costs')
    perf_pred_net = pred_to_perf(df_pred, daily_returns, 0.0012, log=True).rename('Model with transaction costs')

    data = pd.concat([perf_pred_gross, perf_pred_net, 
                      price_to_perf(bench_price[df_pred.index[0]:df_pred.index[-1]], log=True)], axis=1)

    fig = plt.figure(figsize=(14,6))
    plt.title(model, fontsize=14, fontweight='bold')
    sns.lineplot(data=data.rolling(20).mean(), dashes=False)
    plt.show()
    
    
def turnover(df_pred):
    return (df_pred.diff().fillna(0) != 0).any(axis=1).astype(int).sum()


def annual_alpha_plot_grossnet(perf_bench, df_pred, model, daily_returns):
    perf_pred_gross = pred_to_perf(df_pred, daily_returns,  0.).rename('Model without transaction costs')
    perf_pred_net = pred_to_perf(df_pred, daily_returns, 0.0012).rename('Model with transaction costs')

    year_group_bench = perf_bench.resample('Y')
    year_group_pred_gross = perf_pred_gross.resample('Y')
    year_group_pred_net = perf_pred_net.resample('Y')

    annual_returns_bench = (year_group_bench.last() - year_group_bench.first()) / year_group_bench.first()
    annual_returns_pred_gross = (year_group_pred_gross.last() - year_group_pred_gross.first()) / year_group_pred_gross.first()
    annual_returns_pred_net = (year_group_pred_net.last() - year_group_pred_net.first()) / year_group_pred_net.first()

    annual_diff_gross = (annual_returns_pred_gross - annual_returns_bench) * 100
    annual_diff_net = (annual_returns_pred_net - annual_returns_bench) * 100

    df_annual_diff = pd.DataFrame(columns=['year', 'gross', 'net'])
    df_annual_diff['year'] = annual_diff_net.index.year
    df_annual_diff['year'] = df_annual_diff.year.apply(str)
    df_annual_diff['gross'] = annual_diff_gross.values
    df_annual_diff['net'] = annual_diff_net.values
    df_annual_diff = df_annual_diff.melt(id_vars=['year'], var_name='gross_net', value_name='alpha')
    df_annual_diff['sign'] = np.sign(df_annual_diff.alpha)

    fig, ax = plt.subplots(figsize=(14,6))
    sns.barplot(data=df_annual_diff, x='year', y='alpha', hue='gross_net', dodge=True)
    ax.get_legend().set_title(None)
    plt.ylabel('Alpha (%)')
    plt.xlabel('Year')
    plt.title('Annual Return of the model over Benchmark Return : ' + model)
    plt.show()
    
    
def annual_alpha_plot(perf_bench, df_pred, daily_returns):
    
    perf_pred_gross = pred_to_perf(df_pred, daily_returns, 0.).rename('Model without transaction costs')
    
    year_group_bench = perf_bench.resample('Y')
    year_group_pred_gross = perf_pred_gross.resample('Y')
    
    annual_returns_bench = (year_group_bench.last() - year_group_bench.first()) / year_group_bench.first()
    annual_returns_pred_gross = (year_group_pred_gross.last() - year_group_pred_gross.first()) / year_group_pred_gross.first()

    annual_diff_gross = (annual_returns_pred_gross - annual_returns_bench) * 100
    
    df_annual_diff = pd.DataFrame(columns=['year', 'alpha_gross'])
    df_annual_diff['year'] = annual_diff_gross.index.year
    df_annual_diff['year'] = df_annual_diff.year.apply(str)
    df_annual_diff['alpha_gross'] = annual_diff_gross.values
    df_annual_diff['sign_gross'] = np.sign(df_annual_diff.alpha_gross)

    fig, ax = plt.subplots(figsize=(14,6))
    sns.barplot(data=df_annual_diff, x='year', y='alpha_gross', hue='sign_gross', dodge=False)
    ax.get_legend().remove()
    plt.ylabel('Alpha (%)')
    plt.xlabel('Year')
    plt.title('Annual Return of the model (gross) over Benchmark Return : ')
    plot_path = os.path.join(os.path.dirname(__file__)) + 'plots/excess_return.png'
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
    daily_ret = daily_returns[first_date:last_date]
    df_pred_daily = df_pred.reindex(daily_ret.index, method='ffill').shift(1)
    df_perf = (df_pred_daily * daily_ret).sum(axis=1)
    df_cost = (df_pred_daily.diff().fillna(0) != 0).any(axis=1).astype(int) * tax
    if log is False:
        perf = (1 + df_perf - df_cost).cumprod()
    else:
        perf = np.log(1 + df_perf - df_cost).cumsum()
    return perf


def pred_to_daily_ret(df_pred, daily_returns, log=False):
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


def resume_backtest(df_pred_dict, bench_price, price):

    daily_returns = price.pct_change()
    perf_bench = price_to_perf(bench_price.loc[next(iter(df_pred_dict.items()))[1].index[0]:next(iter(df_pred_dict.items()))[1].index[-1]], log=False)

    bench_stats = perf_to_stat(perf_bench, perf_bench)
    stats = []
    stats.append(bench_stats + [0])
    for model_name in df_pred_dict:
        perf_gross = pred_to_perf(df_pred_dict[model_name], daily_returns, 0.)
        perf_net = pred_to_perf(df_pred_dict[model_name], daily_returns, 0.0012)
        
        turnover_num = turnover(df_pred_dict[model_name])
        stats.append(perf_to_stat(perf_gross, perf_net) + [turnover(df_pred_dict[model_name])])
    stats = np.array(stats)

    df_stats = pd.DataFrame(data=stats, columns=['Gross avg. annual return (%)', 'Net avg. annual return (%)', 
                                                'Avg. annual vol. (%)', 'Avg Sharpe ratio', 
                                                'Max. Drawdown (%)', 'Turnover'], 
                            index=['Benchmark SPI'] + list(df_pred_dict.keys())).round(2)

    df_stats.Turnover = df_stats.Turnover.apply(int)

    return df_stats