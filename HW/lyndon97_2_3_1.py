import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
from scipy.stats import norm

ticker = 'LIN'
wd = os.getcwd()
input_dir = wd
ticker_file = os.path.join(input_dir, ticker + '_16_19_new.csv')
plot_dir = os.path.join(input_dir)

try:
    df = pd.read_csv(ticker_file)
    df['Return'] = 100.0 * df['Return']
    pos = len(df[df['Return'] > 0])
    neg = len(df[df['Return'] < 0])

    print("The number of days positive:", pos)
    print("The number of days negative:", neg)

    year1 = '2016'
    year2 = '2019'
    start_date = year1 + '-01-01'
    end_date = year2 + '-12-31'
    df = df[df['Date'] >= start_date]
    df = df[df['Date'] <= end_date]
    low_return = -5
    high_return = 5
    df = df[(df['Return'] > low_return) & (df['Return'] < high_return)]
    fig = plt.figure()
    returns_list = df['Return'].values
    plt.hist(returns_list, density=True, bins=30, label='Daily Returns')
    x = np.linspace(low_return, high_return, 1000)
    # pos = len(df[df['Return'] > 0])
    # neg = len(df[df['Return'] < 0])
    zero = len(df[df['Return'] == 0.0])
    ticker_mean = df['Return'].mean()
    ticker_std = df['Return'].std()

    # plt.plot(x, norm.pdf(x, ticker_mean, ticker_std), color='red',
    #          label='Normal, ' + r'$\mu=$' + str(round(ticker_mean, 2)) +
    #                ', ' + r'$\sigma=$' + str(round(ticker_std, 2)))
    # # plt.title('daily returns for ' + ticker + ' for year ' + year1)
    # plt.legend()
    # output_file = os.path.join(plot_dir, 'returns_' + year1 + '_' + ticker + '_' + str(year2) + '.png')
    # plt.savefig(output_file)
    # plt.show()

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

