import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import calendar
import pandas as pd
import os

def last_month(date):
    year, month = date.year, date.month
    date_month = date.replace(day=calendar.monthrange(year, month)[1])
    return date_month

def exclude_tickers(indice_weight, df_prob):

    df = df_prob.copy()
    df_weight = indice_weight.copy().reindex(df.index, method='ffill')
    for index, row in df.iterrows():
        for ticker, _ in row.items():   
            if pd.isna(df_weight.at[index, ticker]):
                row.at[ticker] = np.nan 
    return df

def clean_tickers(df, ratio=False):
    
    column_names = []
    
    if ratio:
        col_list = df.columns.unique(level=0)
    else:
        col_list = df.columns
        
    for col in col_list:
        column_names.append(col.replace(' SW Equity', '').replace(' VX Equity', ''))
        
    if ratio:
        df.columns = df.columns.set_levels(column_names, level=0)
    else:
        df.columns = column_names


def prob_to_signal(df_prob, indice_weight, n=2, weight=0.1):

    results = exclude_tickers(indice_weight, df_prob)
    df_signal = pd.DataFrame().reindex_like(results)
    for index, row in results.iterrows():
        all_tickers = list(row.index)
        largest_tickers = list(row.nlargest(n).index)
        smallest_tickers = list(row.nsmallest(n).index)
        other_tickers = list(set(all_tickers).difference(largest_tickers + smallest_tickers))
        df_signal.at[index, largest_tickers] = 1
        df_signal.at[index, smallest_tickers] = -1
        df_signal.at[index, other_tickers] = 0.
    df_signal = exclude_tickers(indice_weight, df_signal) * weight
    return df_signal

def prob_to_portfolio(df_prob, indice_weight):

    df_signal = prob_to_signal(df_prob, indice_weight)
    indice_weight = indice_weight.reindex(df_prob.index, method='ffill')
    portfolio = indice_weight + df_signal
    return portfolio


def prob_to_perf(df_prob, indice_weight, daily_returns, tax=0., log=False):

    portfolio = prob_to_portfolio(df_prob, indice_weight)
    daily_returns_backtest = daily_returns.loc[portfolio.index[0]:portfolio.index[-1]]
    perf = portfolio.reindex(daily_returns_backtest.index, method='ffill').mul(daily_returns_backtest).sum(axis=1)
    df_cost = (portfolio.reindex(daily_returns_backtest.index, method='ffill').diff().fillna(0) != 0).any(axis=1).astype(int) * tax

    if log is False:
        perf = (1 + perf - df_cost).cumprod()
    else:
        perf = np.log(1 + perf - df_cost).cumsum()
    return perf

def resume_backtest(df_prob_dict, bench_perf, daily_returns, indice_weight):

    bench_stats = perf_to_stat(bench_perf, bench_perf)
    stats = []
    stats.append(bench_stats + [0])
    for model_name in df_prob_dict:
        perf_gross = prob_to_perf(df_prob_dict[model_name], indice_weight, daily_returns, tax=0.)
        perf_net = prob_to_perf(df_prob_dict[model_name], indice_weight, daily_returns, tax=0.0012)
        
        turnover_num = turnover(df_prob_dict[model_name], indice_weight)
        stats.append(perf_to_stat(perf_gross, perf_net) + [turnover_num])
    stats = np.array(stats)

    df_stats = pd.DataFrame(data=stats, columns=['Gross avg. annual return (%)', 'Net avg. annual return (%)', 
                                                'Avg. annual vol. (%)', 'Avg Sharpe ratio', 
                                                'Max. Drawdown (%)', 'Turnover'], 
                            index=['Benchmark SPI'] + list(df_prob_dict.keys())).round(2)

    df_stats.Turnover = df_stats.Turnover.apply(int)

    return df_stats

def perf_to_stat(perf_gross, perf_net):
    
    # year_group_gross = perf_gross.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0]).mean() * 100
    # year_group_net = perf_net.resample('Y')

    average_year_return_gross = perf_gross.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0]).mean() * 100
    average_year_return_net = perf_net.resample('Y').apply(lambda x: (x[-1] - x[0]) / x[0]).mean() * 100

    # average_year_return_gross = ((year_group_gross.last() - year_group_gross.first()) / year_group_gross.first()).mean() * 100
    # average_year_return_net = ((year_group_net.last() - year_group_net.first()) / year_group_net.first()).mean() * 100

    average_year_std = perf_gross.pct_change().std() * np.sqrt(256) * 100
    average_year_sharpe = average_year_return_net / average_year_std
    
    dd_window = 252
    roll_max = perf_gross.rolling(dd_window).max()
    daily_dd = perf_gross / roll_max - 1
    max_daily_dd = np.abs(daily_dd.rolling(dd_window, min_periods=1).min()).max() * 100

    return [average_year_return_gross, average_year_return_net, average_year_std, average_year_sharpe, max_daily_dd]


def performance_plot(df_prob, model, daily_returns, perf_bench):
    perf_pred_gross = prob_to_perf(df_prob, daily_returns, 0., log=True).rename('Model without transaction costs')
    perf_pred_net = prob_to_perf(df_prob, daily_returns, 0.0012, log=True).rename('Model with transaction costs')

    data = pd.concat([perf_pred_gross, perf_pred_net, perf_bench], axis=1)

    fig = plt.figure(figsize=(14,6))
    plt.title(model, fontsize=14, fontweight='bold')
    sns.lineplot(data=data.rolling(20).mean(), dashes=False)
    plt.show()
    
    
def turnover(df_prob, indice_weight):
    signal = prob_to_signal(df_prob, indice_weight)
    return (signal.diff().fillna(0) != 0).any(axis=1).astype(int).sum()


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
    
    
def annual_alpha_plot(perf_bench, df_prob, daily_returns):
    
    perf_pred_gross = prob_to_perf(df_prob, daily_returns, 0.).rename('Model without transaction costs')
    
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
    plot_path = os.path.join(os.path.dirname(__file__)) + '/plots/excess_return.png'
    plt.savefig(plot_path)
    plt.show()